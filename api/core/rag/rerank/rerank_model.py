from typing import Optional

from core.model_manager import ModelInstance
from core.rag.models.document import Document
from core.rag.rerank.rerank_base import BaseRerankRunner


class RerankModelRunner(BaseRerankRunner):
    """基于重排序模型的文档重排序运行器。

    Attributes:
        rerank_model_instance: 重排序模型实例
    """

    def __init__(self, rerank_model_instance: ModelInstance) -> None:
        """初始化RerankModelRunner。

        Args:
            rerank_model_instance: 重排序模型实例
        """
        self.rerank_model_instance = rerank_model_instance

    def run(
        self,
        query: str,
        documents: list[Document],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> list[Document]:
        """运行重排序模型对文档进行重新排序。

        Args:
            query: 搜索查询字符串
            documents: 待重排序的文档列表
            score_threshold: 分数阈值，低于此值的文档将被过滤
            top_n: 返回的文档数量上限
            user: 用户标识符(可选)

        Returns:
            重排序后的文档列表，按分数降序排列

        Raises:
            ValueError: 如果输入参数无效
        """
        # 预处理文档：去重并收集文档内容
        docs = []
        doc_ids = set()
        unique_documents = []
        for document in documents:
            # 处理来自dify的文档，确保不重复
            if (
                document.provider == "dify"
                and document.metadata is not None
                and document.metadata["doc_id"] not in doc_ids
            ):
                doc_ids.add(document.metadata["doc_id"])
                docs.append(document.page_content)
                unique_documents.append(document)
            # 处理外部文档，确保不重复
            elif document.provider == "external":
                if document not in unique_documents:
                    docs.append(document.page_content)
                    unique_documents.append(document)

        documents = unique_documents

        # 调用重排序模型进行文档重排序
        rerank_result = self.rerank_model_instance.invoke_rerank(
            query=query, docs=docs, score_threshold=score_threshold, top_n=top_n, user=user
        )

        # 处理重排序结果，构建新的文档列表
        rerank_documents = []
        for result in rerank_result.docs:
            # 根据分数阈值过滤文档
            if score_threshold is None or result.score >= score_threshold:
                # 格式化文档
                rerank_document = Document(
                    page_content=result.text,
                    metadata=documents[result.index].metadata,
                    provider=documents[result.index].provider,
                )
                if rerank_document.metadata is not None:
                    rerank_document.metadata["score"] = result.score
                    rerank_documents.append(rerank_document)

        # 按分数降序排序并返回结果
        rerank_documents.sort(key=lambda x: x.metadata.get("score", 0.0), reverse=True)
        return rerank_documents[:top_n] if top_n else rerank_documents
