import math
from collections import Counter
from typing import Optional

import numpy as np

from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler
from core.rag.embedding.cached_embedding import CacheEmbedding
from core.rag.models.document import Document
from core.rag.rerank.entity.weight import VectorSetting, Weights
from core.rag.rerank.rerank_base import BaseRerankRunner


class WeightRerankRunner(BaseRerankRunner):
    """
    基于关键词和向量相似度得分的文档重排序类。
    """

    def __init__(self, tenant_id: str, weights: Weights) -> None:
        """
        初始化WeightRerankRunner。

        :param tenant_id: 租户ID，用于模型管理。
        :param weights: 关键词和向量得分的权重配置。
        """
        self.tenant_id = tenant_id
        self.weights = weights

    def run(
        self,
        query: str,
        documents: list[Document],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> list[Document]:
        """
        运行重排序模型，根据关键词和向量相似度得分对文档进行排序。

        :param query: 搜索查询。
        :param documents: 需要重排序的文档列表。
        :param score_threshold: 得分阈值，低于此阈值的文档将被过滤。
        :param top_n: 返回的文档数量上限。
        :param user: 用户ID，如果需要。

        :return: 重排序后的文档列表。
        """
        # 去除重复文档
        unique_documents = []
        doc_ids = set()
        for document in documents:
            # 根据doc_id去重
            if document.metadata is not None and document.metadata["doc_id"] not in doc_ids:
                doc_ids.add(document.metadata["doc_id"])
                unique_documents.append(document)

        # 生成唯一文档列表
        documents = unique_documents

        # 计算关键词得分和向量相似度得分
        # 1. 基于BM25算法的关键词得分, 响应数据结构为：[float, ...]
        query_scores = self._calculate_keyword_score(query, documents)
        # 2. 基于向量相似度的得分, 响应数据结构为：[float, ...]
        query_vector_scores = self._calculate_cosine(self.tenant_id, query, documents, self.weights.vector_setting)

        # 结合得分并过滤文档
        rerank_documents = []
        for document, query_score, query_vector_score in zip(documents, query_scores, query_vector_scores):
            score = (
                self.weights.vector_setting.vector_weight * query_vector_score
                + self.weights.keyword_setting.keyword_weight * query_score
            )
            if score_threshold and score < score_threshold:
                continue
            if document.metadata is not None:
                document.metadata["score"] = score
                rerank_documents.append(document)

        # 按得分降序排序并返回前top_n个文档
        rerank_documents.sort(key=lambda x: x.metadata["score"] if x.metadata else 0, reverse=True)
        return rerank_documents[:top_n] if top_n else rerank_documents

    def _calculate_keyword_score(self, query: str, documents: list[Document]) -> list[float]:
        """
        计算基于BM25算法的关键词得分。

        :param query: 搜索查询。
        :param documents: 需要得分的文档列表。

        :return: 每个文档的关键词得分列表。
        """
        keyword_table_handler = JiebaKeywordTableHandler()

        # 提取查询关键词
        query_keywords = keyword_table_handler.extract_keywords(query, None)
        # 提取每个文档的关键词
        documents_keywords = []
        for document in documents:
            # 提取文档关键词
            document_keywords = keyword_table_handler.extract_keywords(document.page_content, None)
            if document.metadata is not None:
                document.metadata["keywords"] = document_keywords
                documents_keywords.append(document_keywords)

        # 计算查询关键词的TF（词频）
        query_keyword_counts = Counter(query_keywords)
        total_documents = len(documents)
        all_keywords = set()
        for document_keywords in documents_keywords:
            all_keywords.update(document_keywords)

        # 计算所有关键词的IDF（逆文档频率）
        keyword_idf = {}
        for keyword in all_keywords:
            doc_count_containing_keyword = sum(1 for doc_keywords in documents_keywords if keyword in doc_keywords)
            keyword_idf[keyword] = math.log((1 + total_documents) / (1 + doc_count_containing_keyword)) + 1

        # 计算查询关键词的TF-IDF
        query_tfidf = {}
        """
        query_tfidf 键是关键词，值是该关键词的TF-IDF分数, 所以是一个“关键词->分数”的字典，代表query的TF-IDF向量。

        - 单个关键词的TF-IDF确实是一个小数，但整个query的TF-IDF向量是一个“稀疏向量” ——每个关键词一个分数。
        - 这个向量的每个维度代表一个词，值是该词的TF-IDF分数。
        - 只有在和文档的TF-IDF向量做余弦相似度时，才会把这两个“向量”合成一个相关性分数（小数）。
        """
        for keyword, count in query_keyword_counts.items():
            tf = count
            idf = keyword_idf.get(keyword, 0)
            query_tfidf[keyword] = tf * idf

        # 计算每个文档的TF-IDF
        documents_tfidf = []
        for document_keywords in documents_keywords:
            document_keyword_counts = Counter(document_keywords)
            document_tfidf = {}
            for keyword, count in document_keyword_counts.items():
                tf = count
                idf = keyword_idf.get(keyword, 0)
                document_tfidf[keyword] = tf * idf
            documents_tfidf.append(document_tfidf)

        # 计算余弦相似度
        def cosine_similarity(vec1, vec2):
            """
            计算两个向量的余弦相似度

            参数:
                vec1 (dict): 第一个向量，表示为键值对字典，键是特征名，值是对应特征值
                vec2 (dict): 第二个向量，格式与vec1相同

            返回:
                float: 两个向量的余弦相似度值，范围在[0,1]之间

            先找出query和文档都出现过的关键词（交集）。
            只对这些关键词，把query和文档的TF-IDF分数相乘并求和（点积）。
            分别计算query和文档所有关键词分数的平方和（模长）。
            用点积除以模长的乘积，得到一个0~1之间的小数，表示相关性。

            """
            # 计算两个向量的交集特征
            intersection = set(vec1.keys()) & set(vec2.keys())
            # 计算点积（分子部分）
            numerator = sum(vec1[x] * vec2[x] for x in intersection)

            # 计算每个向量的L2范数平方
            sum1 = sum(vec1[x] ** 2 for x in vec1)
            sum2 = sum(vec2[x] ** 2 for x in vec2)
            # 计算分母（L2范数的乘积）
            denominator = math.sqrt(sum1) * math.sqrt(sum2)

            # 处理分母为零的情况
            if not denominator:
                return 0.0
            else:
                return float(numerator) / denominator

        # 计算每个文档与查询的余弦相似度
        """
        - 在信息检索中，文本的TF-IDF向量就是用“字典”或“数组”来表示的。
        - 余弦相似度的本质是两个向量的夹角余弦值，而这里的“向量”就是“关键词-分数”对的集合。
        """
        similarities = []
        for document_tfidf in documents_tfidf:
            """
            这里的query_tfidf和document_tfidf都是“关键词->分数”的字典，代表两个文本在所有关键词上的“投影”。
            """
            similarity = cosine_similarity(query_tfidf, document_tfidf)
            similarities.append(similarity)

        return similarities

    def _calculate_cosine(
        self, tenant_id: str, query: str, documents: list[Document], vector_setting: VectorSetting
    ) -> list[float]:
        """
        计算查询与文档向量之间的余弦相似度得分。

        :param tenant_id: 租户ID，用于模型管理。
        :param query: 搜索查询。
        :param documents: 需要得分的文档列表。
        :param vector_setting: 向量配置，用于嵌入模型。

        :return: 每个文档的余弦相似度得分列表。
        """
        query_vector_scores = []

        model_manager = ModelManager()
        embedding_model = model_manager.get_model_instance(
            tenant_id=tenant_id,
            provider=vector_setting.embedding_provider_name,
            model_type=ModelType.TEXT_EMBEDDING,
            model=vector_setting.embedding_model_name,
        )
        cache_embedding = CacheEmbedding(embedding_model)
        query_vector = cache_embedding.embed_query(query)

        # 计算每个文档的余弦相似度
        for document in documents:
            if document.metadata and "score" in document.metadata:
                query_vector_scores.append(document.metadata["score"])
            else:
                vec1 = np.array(query_vector)
                vec2 = np.array(document.vector)
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                cosine_sim = dot_product / (norm_vec1 * norm_vec2)
                query_vector_scores.append(cosine_sim)

        return query_vector_scores
