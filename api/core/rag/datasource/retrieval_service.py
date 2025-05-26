import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from flask import Flask, current_app
from sqlalchemy.orm import load_only

from configs import dify_config
from core.rag.data_post_processor.data_post_processor import DataPostProcessor
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.embedding.retrieval import RetrievalSegments
from core.rag.entities.metadata_entities import MetadataCondition
from core.rag.index_processor.constant.index_type import IndexType
from core.rag.models.document import Document
from core.rag.rerank.rerank_type import RerankMode
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from extensions.ext_database import db
from models.dataset import ChildChunk, Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument
from services.external_knowledge_service import ExternalDatasetService

# 默认检索模型配置
# Args:
#   search_method: 检索方法，默认为语义搜索
#   reranking_enable: 是否启用重排序，默认为False
#   reranking_model: 重排序模型配置
#   top_k: 返回结果数量，默认为2
#   score_threshold_enabled: 是否启用分数阈值过滤，默认为False
default_retrieval_model = {
    "search_method": RetrievalMethod.SEMANTIC_SEARCH.value,
    "reranking_enable": False,
    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    "top_k": 2,
    "score_threshold_enabled": False,
}


class RetrievalService:
    """检索服务类，提供多种检索方法

    检索方法：支持多种检索方式，包括：
        关键词搜索 (keyword_search)：基于关键词匹配进行检索。
        语义搜索 (embedding_search)：基于向量相似度进行检索。
        全文搜索 (full_text_index_search)：基于全文索引进行检索。
        混合搜索 (HYBRID_SEARCH)：结合多种检索方法的结果。

    多线程优化：
        使用 ThreadPoolExecutor 并行执行不同的检索方法，以提高检索效率。
    重排序：
        支持对检索结果进行重排序，提升结果的相关性。
    外部知识库检索：
        支持从外部知识库中检索数据。
    """

    # Cache precompiled regular expressions to avoid repeated compilation
    @classmethod
    def retrieve(
        cls,
        retrieval_method: str,
        dataset_id: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float] = 0.0,
        reranking_model: Optional[dict] = None,
        reranking_mode: str = "reranking_model",
        weights: Optional[dict] = None,
        document_ids_filter: Optional[list[str]] = None,
    ):
        """执行检索操作

        Args:
            cls: 类对象，用于调用类方法。
            retrieval_method: 检索方法（如关键词搜索、语义搜索等）
            dataset_id: 指定要检索的数据集
            query: 查询字符串
            top_k: 返回的文档数量上限
            score_threshold: 分数阈值，用于过滤低分文档
            reranking_model: 重排序模型配置，可选
            reranking_mode: 重排序模式，默认为"reranking_model"
            weights: 权重，用于重排序时的加权计算，可选
            document_ids_filter: 文档ID过滤器，可选

        Returns:
            list[Document]: 检索结果文档列表

        Raises:
            ValueError: 如果检索过程中出现异常
        """
        if not query:
            return []
        dataset = cls._get_dataset(dataset_id)
        if not dataset:
            return []

        all_documents: list[Document] = []
        # 异常处理：每个检索方法都会捕获异常，并将异常信息记录到 exceptions 列表中，最终统一抛出
        exceptions: list[str] = []

        # 使用 ThreadPoolExecutor 并行执行检索任务，最大线程数由配置 dify_config.RETRIEVAL_SERVICE_EXECUTORS 决定。
        with ThreadPoolExecutor(max_workers=dify_config.RETRIEVAL_SERVICE_EXECUTORS) as executor:  # type: ignore
            futures = []
            if retrieval_method == "keyword_search":
                futures.append(
                    executor.submit(
                        cls.keyword_search,  # 关键词搜索
                        flask_app=current_app._get_current_object(),  # type: ignore
                        dataset_id=dataset_id,
                        query=query,
                        top_k=top_k,
                        all_documents=all_documents,
                        exceptions=exceptions,
                        document_ids_filter=document_ids_filter,
                    )
                )
            if RetrievalMethod.is_support_semantic_search(retrieval_method):
                futures.append(
                    executor.submit(
                        cls.embedding_search,  # 向量嵌入搜索
                        flask_app=current_app._get_current_object(),  # type: ignore
                        dataset_id=dataset_id,
                        query=query,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        reranking_model=reranking_model,
                        all_documents=all_documents,
                        retrieval_method=retrieval_method,
                        exceptions=exceptions,
                        document_ids_filter=document_ids_filter,
                    )
                )
            if RetrievalMethod.is_support_fulltext_search(retrieval_method):
                futures.append(
                    executor.submit(
                        cls.full_text_index_search,  # 执行全文索引搜索
                        flask_app=current_app._get_current_object(),  # type: ignore
                        dataset_id=dataset_id,
                        query=query,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        reranking_model=reranking_model,
                        all_documents=all_documents,
                        retrieval_method=retrieval_method,
                        exceptions=exceptions,
                        document_ids_filter=document_ids_filter,
                    )
                )
            concurrent.futures.wait(futures, timeout=30, return_when=concurrent.futures.ALL_COMPLETED)

        if exceptions:
            raise ValueError(";\n".join(exceptions))

        if retrieval_method == RetrievalMethod.HYBRID_SEARCH.value:
            # 重排序：通过 DataPostProcessor 类对检索结果进行重排序，支持自定义重排序模型和权重。
            data_post_processor = DataPostProcessor(
                str(dataset.tenant_id), reranking_mode, reranking_model, weights, False
            )
            all_documents = data_post_processor.invoke(
                query=query,
                documents=all_documents,
                score_threshold=score_threshold,
                top_n=top_k,
            )

        return all_documents

    @classmethod
    def external_retrieve(
        cls,
        dataset_id: str,
        query: str,
        external_retrieval_model: Optional[dict] = None,
        metadata_filtering_conditions: Optional[dict] = None,
    ):
        """执行外部知识检索

        Args:
            dataset_id: 数据集ID
            query: 查询字符串
            external_retrieval_model: 外部检索模型配置，可选
            metadata_filtering_conditions: 元数据过滤条件，可选

        Returns:
            list: 检索结果列表
        """
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return []
        metadata_condition = (
            MetadataCondition(**metadata_filtering_conditions) if metadata_filtering_conditions else None
        )
        all_documents = ExternalDatasetService.fetch_external_knowledge_retrieval(
            dataset.tenant_id,
            dataset_id,
            query,
            external_retrieval_model or {},
            metadata_condition=metadata_condition,
        )
        return all_documents

    @classmethod
    def _get_dataset(cls, dataset_id: str) -> Optional[Dataset]:
        """根据数据集ID获取数据集对象

        Args:
            dataset_id: 数据集ID

        Returns:
            Optional[Dataset]: 数据集对象，如果不存在则返回None
        """
        return db.session.query(Dataset).filter(Dataset.id == dataset_id).first()

    @classmethod
    def keyword_search(
        cls,
        flask_app: Flask,
        dataset_id: str,
        query: str,
        top_k: int,
        all_documents: list,
        exceptions: list,
        document_ids_filter: Optional[list[str]] = None,
    ):
        """执行关键词搜索

        Args:
            flask_app: Flask应用实例
            dataset_id: 数据集ID
            query: 查询字符串
            top_k: 返回结果数量
            all_documents: 用于存储检索结果的文档列表
            exceptions: 用于存储异常的列表
            document_ids_filter: 文档ID过滤器，可选
        """
        """
        在Flask应用的上下文中执行数据集查询和关键词搜索操作。
        这样做可以确保在应用的上下文中正确地处理请求，并能够访问Flask应用中的资源。
        """
        with flask_app.app_context():
            try:
                # 尝试根据数据集ID获取数据集实例
                dataset = cls._get_dataset(dataset_id)
                # 如果数据集不存在，则抛出ValueError异常
                if not dataset:
                    raise ValueError("dataset not found")

                # 创建Keyword实例，关联到获取的数据集
                keyword = Keyword(dataset=dataset)

                # 执行关键词搜索，并获取搜索结果
                documents = keyword.search(
                    cls.escape_query_for_search(query), top_k=top_k, document_ids_filter=document_ids_filter
                )
                # 将搜索到的文档添加到all_documents列表中
                all_documents.extend(documents)
            # 捕获并处理在try块中抛出的任何异常
            except Exception as e:
                # 将异常信息转换为字符串并添加到exceptions列表中
                exceptions.append(str(e))


    @classmethod
    def embedding_search(
        cls,
        flask_app: Flask,
        dataset_id: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
        reranking_model: Optional[dict],
        all_documents: list,
        retrieval_method: str,
        exceptions: list,
        document_ids_filter: Optional[list[str]] = None,
    ):
        """执行向量嵌入搜索

        Args:
            flask_app: Flask应用实例
            dataset_id: 数据集ID
            query: 查询字符串
            top_k: 返回结果数量
            score_threshold: 分数阈值，可选
            reranking_model: 重排序模型配置，可选
            all_documents: 用于存储检索结果的文档列表
            retrieval_method: 检索方法
            exceptions: 用于存储异常的列表
            document_ids_filter: 文档ID过滤器，可选
        """
        with flask_app.app_context():
            try:
                dataset = cls._get_dataset(dataset_id)
                if not dataset:
                    raise ValueError("dataset not found")

                vector = Vector(dataset=dataset)
                documents = vector.search_by_vector(
                    query,
                    search_type="similarity_score_threshold",
                    top_k=top_k,
                    score_threshold=score_threshold,
                    filter={"group_id": [dataset.id]},
                    document_ids_filter=document_ids_filter,
                )

                if documents:
                    if (
                        reranking_model
                        and reranking_model.get("reranking_model_name")
                        and reranking_model.get("reranking_provider_name")
                        and retrieval_method == RetrievalMethod.SEMANTIC_SEARCH.value
                    ):
                        data_post_processor = DataPostProcessor(
                            str(dataset.tenant_id), str(RerankMode.RERANKING_MODEL.value), reranking_model, None, False
                        )
                        all_documents.extend(
                            data_post_processor.invoke(
                                query=query,
                                documents=documents,
                                score_threshold=score_threshold,
                                top_n=len(documents),
                            )
                        )
                    else:
                        all_documents.extend(documents)
            except Exception as e:
                exceptions.append(str(e))

    @classmethod
    def full_text_index_search(
        cls,
        flask_app: Flask,
        dataset_id: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
        reranking_model: Optional[dict],
        all_documents: list,
        retrieval_method: str,
        exceptions: list,
        document_ids_filter: Optional[list[str]] = None,
    ):
        """执行全文索引搜索

        Args:
            flask_app: Flask应用实例
            dataset_id: 数据集ID
            query: 查询字符串
            top_k: 返回结果数量
            score_threshold: 分数阈值，可选
            reranking_model: 重排序模型配置，可选
            all_documents: 用于存储检索结果的文档列表
            retrieval_method: 检索方法
            exceptions: 用于存储异常的列表
            document_ids_filter: 文档ID过滤器，可选
        """
        with flask_app.app_context():
            try:
                dataset = cls._get_dataset(dataset_id)
                if not dataset:
                    raise ValueError("dataset not found")

                vector_processor = Vector(dataset=dataset)

                documents = vector_processor.search_by_full_text(
                    cls.escape_query_for_search(query), top_k=top_k, document_ids_filter=document_ids_filter
                )
                if documents:
                    if (
                        reranking_model
                        and reranking_model.get("reranking_model_name")
                        and reranking_model.get("reranking_provider_name")
                        and retrieval_method == RetrievalMethod.FULL_TEXT_SEARCH.value
                    ):
                        data_post_processor = DataPostProcessor(
                            str(dataset.tenant_id), str(RerankMode.RERANKING_MODEL.value), reranking_model, None, False
                        )
                        all_documents.extend(
                            data_post_processor.invoke(
                                query=query,
                                documents=documents,
                                score_threshold=score_threshold,
                                top_n=len(documents),
                            )
                        )
                    else:
                        all_documents.extend(documents)
            except Exception as e:
                exceptions.append(str(e))

    @staticmethod
    def escape_query_for_search(query: str) -> str:
        """转义查询字符串中的特殊字符

        Args:
            query: 原始查询字符串

        Returns:
            str: 转义后的查询字符串
        """
        return query.replace('"', '\\"')

    @classmethod
    def format_retrieval_documents(cls, documents: list[Document]) -> list[RetrievalSegments]:
        """格式化检索文档

        使用优化的批处理方式格式化检索文档，处理父子文档关系

        Args:
            documents: 原始文档列表

        Returns:
            list[RetrievalSegments]: 格式化后的检索结果列表

        Raises:
            Exception: 如果处理过程中出现错误
        """
        if not documents:
            return []

        try:
            # Collect document IDs
            document_ids = {doc.metadata.get("document_id") for doc in documents if "document_id" in doc.metadata}
            if not document_ids:
                return []

            # Batch query dataset documents
            dataset_documents = {
                doc.id: doc
                for doc in db.session.query(DatasetDocument)
                .filter(DatasetDocument.id.in_(document_ids))
                .options(load_only(DatasetDocument.id, DatasetDocument.doc_form, DatasetDocument.dataset_id))
                .all()
            }

            records = []
            include_segment_ids = set()
            segment_child_map = {}

            # Process documents
            for document in documents:
                document_id = document.metadata.get("document_id")
                if document_id not in dataset_documents:
                    continue

                dataset_document = dataset_documents[document_id]
                if not dataset_document:
                    continue

                if dataset_document.doc_form == IndexType.PARENT_CHILD_INDEX:
                    # Handle parent-child documents
                    child_index_node_id = document.metadata.get("doc_id")

                    child_chunk = (
                        db.session.query(ChildChunk).filter(ChildChunk.index_node_id == child_index_node_id).first()
                    )

                    if not child_chunk:
                        continue

                    segment = (
                        db.session.query(DocumentSegment)
                        .filter(
                            DocumentSegment.dataset_id == dataset_document.dataset_id,
                            DocumentSegment.enabled == True,
                            DocumentSegment.status == "completed",
                            DocumentSegment.id == child_chunk.segment_id,
                        )
                        .options(
                            load_only(
                                DocumentSegment.id,
                                DocumentSegment.content,
                                DocumentSegment.answer,
                            )
                        )
                        .first()
                    )

                    if not segment:
                        continue

                    if segment.id not in include_segment_ids:
                        include_segment_ids.add(segment.id)
                        child_chunk_detail = {
                            "id": child_chunk.id,
                            "content": child_chunk.content,
                            "position": child_chunk.position,
                            "score": document.metadata.get("score", 0.0),
                        }
                        map_detail = {
                            "max_score": document.metadata.get("score", 0.0),
                            "child_chunks": [child_chunk_detail],
                        }
                        segment_child_map[segment.id] = map_detail
                        record = {
                            "segment": segment,
                        }
                        records.append(record)
                    else:
                        child_chunk_detail = {
                            "id": child_chunk.id,
                            "content": child_chunk.content,
                            "position": child_chunk.position,
                            "score": document.metadata.get("score", 0.0),
                        }
                        segment_child_map[segment.id]["child_chunks"].append(child_chunk_detail)
                        segment_child_map[segment.id]["max_score"] = max(
                            segment_child_map[segment.id]["max_score"], document.metadata.get("score", 0.0)
                        )
                else:
                    # Handle normal documents
                    index_node_id = document.metadata.get("doc_id")
                    if not index_node_id:
                        continue

                    segment = (
                        db.session.query(DocumentSegment)
                        .filter(
                            DocumentSegment.dataset_id == dataset_document.dataset_id,
                            DocumentSegment.enabled == True,
                            DocumentSegment.status == "completed",
                            DocumentSegment.index_node_id == index_node_id,
                        )
                        .first()
                    )

                    if not segment:
                        continue

                    include_segment_ids.add(segment.id)
                    record = {
                        "segment": segment,
                        "score": document.metadata.get("score"),  # type: ignore
                    }
                    records.append(record)

            # Add child chunks information to records
            for record in records:
                if record["segment"].id in segment_child_map:
                    record["child_chunks"] = segment_child_map[record["segment"].id].get("child_chunks")  # type: ignore
                    record["score"] = segment_child_map[record["segment"].id]["max_score"]

            return [RetrievalSegments(**record) for record in records]
        except Exception as e:
            db.session.rollback()
            raise e
