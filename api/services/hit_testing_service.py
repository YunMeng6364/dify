import logging
import time
from typing import Any

from core.rag.datasource.retrieval_service import RetrievalService
from core.rag.models.document import Document
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from extensions.ext_database import db
from models.account import Account
from models.dataset import Dataset, DatasetQuery

# 默认检索模型配置
# - search_method: 检索方法，默认为语义搜索
# - reranking_enable: 是否启用重排序
# - reranking_model: 重排序模型配置
# - top_k: 返回结果数量
# - score_threshold_enabled: 是否启用分数阈值

default_retrieval_model = {
    "search_method": RetrievalMethod.SEMANTIC_SEARCH.value,
    "reranking_enable": False,
    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    "top_k": 2,
    "score_threshold_enabled": False,
}


class HitTestingService:
    """命中测试服务类，负责处理数据集命中测试相关的检索逻辑。

    主要功能包括:
    - 执行检索操作
    - 处理外部知识检索
    - 格式化检索结果
    - 参数校验
    """

    @classmethod
    def retrieve(
        cls,
        dataset: Dataset,
        query: str,
        account: Account,
        retrieval_model: Any,  # FIXME drop this any
        external_retrieval_model: dict,
        limit: int = 10,
    ) -> dict:
        """执行检索操作

        Args:
            dataset: 数据集对象
            query: 查询字符串
            account: 账户对象
            retrieval_model: 检索模型配置
            external_retrieval_model: 外部检索模型配置
            limit: 返回结果限制数量，默认为10

        Returns:
            dict: 包含查询内容和检索结果的字典

        Raises:
            Exception: 检索过程中可能抛出的各种异常
        """
        start = time.perf_counter()

        # 获取检索模型配置，如果没有设置则使用默认配置
        if not retrieval_model:
            retrieval_model = dataset.retrieval_model or default_retrieval_model

        # 核心逻辑 调用检索服务执行实际检索操作
        all_documents = RetrievalService.retrieve(
            retrieval_method=retrieval_model.get("search_method", "semantic_search"),
            dataset_id=dataset.id,
            query=query,
            top_k=retrieval_model.get("top_k", 2),
            score_threshold=retrieval_model.get("score_threshold", 0.0)
            if retrieval_model["score_threshold_enabled"]
            else 0.0,
            reranking_model=retrieval_model.get("reranking_model", None)
            if retrieval_model["reranking_enable"]
            else None,
            reranking_mode=retrieval_model.get("reranking_mode") or "reranking_model",
            weights=retrieval_model.get("weights", None),
        )

        end = time.perf_counter()
        logging.debug(f"Hit testing retrieve in {end - start:0.4f} seconds")

        # 创建并保存数据集查询记录
        dataset_query = DatasetQuery(
            dataset_id=dataset.id, content=query, source="hit_testing", created_by_role="account", created_by=account.id
        )

        db.session.add(dataset_query)
        db.session.commit()

        return cls.compact_retrieve_response(query, all_documents)  # type: ignore

    @classmethod
    def external_retrieve(
        cls,
        dataset: Dataset,
        query: str,
        account: Account,
        external_retrieval_model: dict,
        metadata_filtering_conditions: dict,
    ) -> dict:
        """执行外部知识检索

        Args:
            dataset: 数据集对象
            query: 查询字符串
            account: 账户对象
            external_retrieval_model: 外部检索模型配置
            metadata_filtering_conditions: 元数据过滤条件

        Returns:
            dict: 包含查询内容和检索结果的字典

        Raises:
            Exception: 检索过程中可能抛出的各种异常
        """
        if dataset.provider != "external":
            return {
                "query": {"content": query},
                "records": [],
            }

        start = time.perf_counter()

        all_documents = RetrievalService.external_retrieve(
            dataset_id=dataset.id,
            query=cls.escape_query_for_search(query),
            external_retrieval_model=external_retrieval_model,
            metadata_filtering_conditions=metadata_filtering_conditions,
        )

        end = time.perf_counter()
        logging.debug(f"External knowledge hit testing retrieve in {end - start:0.4f} seconds")

        dataset_query = DatasetQuery(
            dataset_id=dataset.id, content=query, source="hit_testing", created_by_role="account", created_by=account.id
        )

        db.session.add(dataset_query)
        db.session.commit()

        return dict(cls.compact_external_retrieve_response(dataset, query, all_documents))

    @classmethod
    def compact_retrieve_response(cls, query: str, documents: list[Document]):
        """格式化检索结果

        Args:
            query: 查询字符串
            documents: 文档列表

        Returns:
            dict: 格式化后的检索结果
        """
        records = RetrievalService.format_retrieval_documents(documents)

        return {
            "query": {
                "content": query,
            },
            "records": [record.model_dump() for record in records],
        }

    @classmethod
    def compact_external_retrieve_response(cls, dataset: Dataset, query: str, documents: list) -> dict[Any, Any]:
        """格式化外部知识检索结果

        Args:
            dataset: 数据集对象
            query: 查询字符串
            documents: 文档列表

        Returns:
            dict: 格式化后的外部知识检索结果
        """
        records = []
        if dataset.provider == "external":
            for document in documents:
                record = {
                    "content": document.get("content", None),
                    "title": document.get("title", None),
                    "score": document.get("score", None),
                    "metadata": document.get("metadata", None),
                }
                records.append(record)
            return {
                "query": {"content": query},
                "records": records,
            }
        return {"query": {"content": query}, "records": []}

    @classmethod
    def hit_testing_args_check(cls, args):
        """命中测试参数校验

        Args:
            args: 参数字典

        Raises:
            ValueError: 当查询参数无效时抛出
        """
        query = args["query"]

        if not query or len(query) > 250:
            raise ValueError("Query is required and cannot exceed 250 characters")

    @staticmethod
    def escape_query_for_search(query: str) -> str:
        """转义查询字符串中的特殊字符

        Args:
            query: 原始查询字符串

        Returns:
            str: 转义后的查询字符串
        """
        return query.replace('"', '\\"')
