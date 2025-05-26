import json
from collections import defaultdict
from typing import Any, Optional

from pydantic import BaseModel

from configs import dify_config
from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler
from core.rag.datasource.keyword.keyword_base import BaseKeyword
from core.rag.models.document import Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from models.dataset import Dataset, DatasetKeywordTable, DocumentSegment


class KeywordTableConfig(BaseModel):
    """关键词表配置模型

    Attributes:
        max_keywords_per_chunk (int): 每个文本块的最大关键词数量，默认为10
    """
    max_keywords_per_chunk: int = 10


class Jieba(BaseKeyword):
    """基于结巴分词的关键词处理类

    继承自BaseKeyword，提供关键词提取、存储和搜索功能

    Attributes:
        dataset (Dataset): 关联的数据集对象
        _config (KeywordTableConfig): 关键词表配置对象
    """

    def __init__(self, dataset: Dataset):
        """初始化Jieba关键词处理器

        Args:
            dataset: 要处理的数据集对象
        """
        super().__init__(dataset)
        self._config = KeywordTableConfig()

    def create(self, texts: list[Document], **kwargs) -> BaseKeyword:
        """创建关键词表并索引所有文本

        Args:
            texts: 要索引的文档列表
            **kwargs: 额外参数

        Returns:
            返回自身实例以便链式调用

        处理流程:
            1. 获取分布式锁防止并发操作
            2. 提取每个文档的关键词
            3. 更新文档段的关键词信息
            4. 将关键词添加到关键词表
            5. 保存关键词表
        """
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            keyword_table_handler = JiebaKeywordTableHandler()
            keyword_table = self._get_dataset_keyword_table()
            for text in texts:
                keywords = keyword_table_handler.extract_keywords(
                    text.page_content, self._config.max_keywords_per_chunk
                )
                if text.metadata is not None:
                    self._update_segment_keywords(self.dataset.id, text.metadata["doc_id"], list(keywords))
                    keyword_table = self._add_text_to_keyword_table(
                        keyword_table or {}, text.metadata["doc_id"], list(keywords)
                    )

            self._save_dataset_keyword_table(keyword_table)

            return self

    def add_texts(self, texts: list[Document], **kwargs):
        """向关键词表中添加新文本

        Args:
            texts: 要添加的文档列表
            **kwargs: 可包含预定义的关键词列表(keywords_list)

        处理流程:
            1. 获取分布式锁
            2. 提取或使用预定义的关键词
            3. 更新文档段信息
            4. 将关键词添加到关键词表
        """
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            keyword_table_handler = JiebaKeywordTableHandler()

            keyword_table = self._get_dataset_keyword_table()
            keywords_list = kwargs.get("keywords_list")
            for i in range(len(texts)):
                text = texts[i]
                if keywords_list:
                    keywords = keywords_list[i]
                    if not keywords:
                        keywords = keyword_table_handler.extract_keywords(
                            text.page_content, self._config.max_keywords_per_chunk
                        )
                else:
                    keywords = keyword_table_handler.extract_keywords(
                        text.page_content, self._config.max_keywords_per_chunk
                    )
                if text.metadata is not None:
                    self._update_segment_keywords(self.dataset.id, text.metadata["doc_id"], list(keywords))
                    keyword_table = self._add_text_to_keyword_table(
                        keyword_table or {}, text.metadata["doc_id"], list(keywords)
                    )

            self._save_dataset_keyword_table(keyword_table)

    def text_exists(self, id: str) -> bool:
        """检查指定ID的文本是否存在于关键词表中

        Args:
            id: 要检查的文本ID

        Returns:
            如果存在返回True，否则返回False
        """
        keyword_table = self._get_dataset_keyword_table()
        if keyword_table is None:
            return False
        return id in set.union(*keyword_table.values())

    def delete_by_ids(self, ids: list[str]) -> None:
        """根据ID列表删除关键词表中的记录

        Args:
            ids: 要删除的文本ID列表

        处理流程:
            1. 获取分布式锁
            2. 从关键词表中删除指定ID
            3. 保存更新后的关键词表
        """
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            keyword_table = self._get_dataset_keyword_table()
            if keyword_table is not None:
                keyword_table = self._delete_ids_from_keyword_table(keyword_table, ids)

            self._save_dataset_keyword_table(keyword_table)

    def search(self, query: str, **kwargs: Any) -> list[Document]:
        """根据查询字符串搜索相关文档

        Args:
            query: 查询字符串
            **kwargs: 可包含:
                top_k: 返回的文档数量，默认为4
                document_ids_filter: 文档ID过滤器

        Returns:
            匹配的文档列表，按相关性排序
        """

        # 1. 获取关键词表
        keyword_table = self._get_dataset_keyword_table()

        # 2. 获取搜索参数
        k = kwargs.get("top_k", 4)
        document_ids_filter = kwargs.get("document_ids_filter")

        # 3. 根据查询检索相关文本ID
        sorted_chunk_indices = self._retrieve_ids_by_query(keyword_table or {}, query, k)

        # 4. 从数据库获取文档内容
        documents = []
        for chunk_index in sorted_chunk_indices:
            # 构建数据库查询
            segment_query = db.session.query(DocumentSegment).filter(
                DocumentSegment.dataset_id == self.dataset.id,
                DocumentSegment.index_node_id == chunk_index
            )

            # 应用文档ID过滤
            if document_ids_filter:
                segment_query = segment_query.filter(
                    DocumentSegment.document_id.in_(document_ids_filter)
                )

            # 获取文档段
            segment = segment_query.first()

            if segment:
                # 创建Document对象
                documents.append(
                    Document(
                        page_content=segment.content,
                        metadata={
                            "doc_id"     : chunk_index,
                            "doc_hash"   : segment.index_node_hash,
                            "document_id": segment.document_id,
                            "dataset_id" : segment.dataset_id,
                        },
                    )
                )

        return documents

    def delete(self) -> None:
        """删除整个关键词表

        处理流程:
            1. 获取分布式锁
            2. 从数据库删除关键词表记录
            3. 如果是文件存储，则删除对应的文件
        """
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            dataset_keyword_table = self.dataset.dataset_keyword_table
            if dataset_keyword_table:
                db.session.delete(dataset_keyword_table)
                db.session.commit()
                if dataset_keyword_table.data_source_type != "database":
                    file_key = "keyword_files/" + self.dataset.tenant_id + "/" + self.dataset.id + ".txt"
                    storage.delete(file_key)

    def _save_dataset_keyword_table(self, keyword_table):
        """保存关键词表到数据库或文件系统

        Args:
            keyword_table: 要保存的关键词表字典
        """
        keyword_table_dict = {
            "__type__": "keyword_table",
            "__data__": {"index_id": self.dataset.id, "summary": None, "table": keyword_table},
        }
        dataset_keyword_table = self.dataset.dataset_keyword_table
        keyword_data_source_type = dataset_keyword_table.data_source_type
        if keyword_data_source_type == "database":
            dataset_keyword_table.keyword_table = json.dumps(keyword_table_dict, cls=SetEncoder)
            db.session.commit()
        else:
            file_key = "keyword_files/" + self.dataset.tenant_id + "/" + self.dataset.id + ".txt"
            if storage.exists(file_key):
                storage.delete(file_key)
            storage.save(file_key, json.dumps(keyword_table_dict, cls=SetEncoder).encode("utf-8"))

    def _get_dataset_keyword_table(self) -> Optional[dict]:
        """获取数据集的关键词表

        Returns:
            关键词表字典，如果不存在则创建并返回空字典
        """
        dataset_keyword_table = self.dataset.dataset_keyword_table
        if dataset_keyword_table:
            keyword_table_dict = dataset_keyword_table.keyword_table_dict
            if keyword_table_dict:
                return dict(keyword_table_dict["__data__"]["table"])
        else:
            keyword_data_source_type = dify_config.KEYWORD_DATA_SOURCE_TYPE
            dataset_keyword_table = DatasetKeywordTable(
                dataset_id=self.dataset.id,
                keyword_table="",
                data_source_type=keyword_data_source_type,
            )
            if keyword_data_source_type == "database":
                dataset_keyword_table.keyword_table = json.dumps(
                    {
                        "__type__": "keyword_table",
                        "__data__": {"index_id": self.dataset.id, "summary": None, "table": {}},
                    },
                    cls=SetEncoder,
                )
            db.session.add(dataset_keyword_table)
            db.session.commit()

        return {}

    def _add_text_to_keyword_table(self, keyword_table: dict, id: str, keywords: list[str]) -> dict:
        """将文本及其关键词添加到关键词表

        Args:
            keyword_table: 当前关键词表
            id: 文本ID
            keywords: 关键词列表

        Returns:
            更新后的关键词表
        """
        for keyword in keywords:
            if keyword not in keyword_table:
                keyword_table[keyword] = set()
            keyword_table[keyword].add(id)
        return keyword_table

    def _delete_ids_from_keyword_table(self, keyword_table: dict, ids: list[str]) -> dict:
        """从关键词表中删除指定的文本ID

        Args:
            keyword_table: 当前关键词表
            ids: 要删除的文本ID列表

        Returns:
            清理后的关键词表
        """
        node_idxs_to_delete = set(ids)

        keywords_to_delete = set()
        for keyword, node_idxs in keyword_table.items():
            if node_idxs_to_delete.intersection(node_idxs):
                keyword_table[keyword] = node_idxs.difference(node_idxs_to_delete)
                if not keyword_table[keyword]:
                    keywords_to_delete.add(keyword)

        for keyword in keywords_to_delete:
            del keyword_table[keyword]

        return keyword_table

    def _retrieve_ids_by_query(self, keyword_table: dict, query: str, k: int = 4):
        """根据查询字符串检索相关文本ID

        Args:
            keyword_table: 关键词表
            query: 查询字符串
            k: 返回的最大结果数

        Returns:
            按相关性排序的文本ID列表
        """
        keyword_table_handler = JiebaKeywordTableHandler()
        keywords = keyword_table_handler.extract_keywords(query)

        chunk_indices_count: dict[str, int] = defaultdict(int)
        keywords_list = [keyword for keyword in keywords if keyword in set(keyword_table.keys())]
        for keyword in keywords_list:
            for node_id in keyword_table[keyword]:
                chunk_indices_count[node_id] += 1

        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )

        return sorted_chunk_indices[:k]

    def _update_segment_keywords(self, dataset_id: str, node_id: str, keywords: list[str]):
        """更新文档段的关键词信息

        Args:
            dataset_id: 数据集ID
            node_id: 文档段ID
            keywords: 关键词列表
        """
        document_segment = (
            db.session.query(DocumentSegment)
            .filter(DocumentSegment.dataset_id == dataset_id, DocumentSegment.index_node_id == node_id)
            .first()
        )
        if document_segment:
            document_segment.keywords = keywords
            db.session.add(document_segment)
            db.session.commit()

    def create_segment_keywords(self, node_id: str, keywords: list[str]):
        """创建文档段的关键词索引

        Args:
            node_id: 文档段ID
            keywords: 关键词列表
        """
        keyword_table = self._get_dataset_keyword_table()
        self._update_segment_keywords(self.dataset.id, node_id, keywords)
        keyword_table = self._add_text_to_keyword_table(keyword_table or {}, node_id, keywords)
        self._save_dataset_keyword_table(keyword_table)

    def multi_create_segment_keywords(self, pre_segment_data_list: list):
        """批量创建文档段的关键词索引

        Args:
            pre_segment_data_list: 预处理的文档段数据列表，每个元素包含segment和keywords
        """
        keyword_table_handler = JiebaKeywordTableHandler()
        keyword_table = self._get_dataset_keyword_table()
        for pre_segment_data in pre_segment_data_list:
            segment = pre_segment_data["segment"]
            if pre_segment_data["keywords"]:
                segment.keywords = pre_segment_data["keywords"]
                keyword_table = self._add_text_to_keyword_table(
                    keyword_table or {}, segment.index_node_id, pre_segment_data["keywords"]
                )
            else:
                keywords = keyword_table_handler.extract_keywords(segment.content, self._config.max_keywords_per_chunk)
                segment.keywords = list(keywords)
                keyword_table = self._add_text_to_keyword_table(
                    keyword_table or {}, segment.index_node_id, list(keywords)
                )
        self._save_dataset_keyword_table(keyword_table)

    def update_segment_keywords_index(self, node_id: str, keywords: list[str]):
        """更新文档段的关键词索引

        Args:
            node_id: 文档段ID
            keywords: 新的关键词列表
        """
        keyword_table = self._get_dataset_keyword_table()
        keyword_table = self._add_text_to_keyword_table(keyword_table or {}, node_id, keywords)
        self._save_dataset_keyword_table(keyword_table)


class SetEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理集合类型

    将集合类型转换为列表以便JSON序列化
    """

    def default(self, obj):
        """重写默认编码方法

        Args:
            obj: 要编码的对象

        Returns:
            编码后的对象
        """
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)
