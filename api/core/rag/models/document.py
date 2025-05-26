from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel


class ChildDocument(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str

    vector: Optional[list[float]] = None

    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    metadata: dict = {}


class Document(BaseModel):
    """
    用于存储一段文本内容及其相关元数据的类。

    Attributes:
        page_content (str): 文档的主要文本内容。
        vector (Optional[list[float]]): 文档的向量表示，默认为 None。
        metadata (dict): 与文档相关的任意元数据（例如来源、与其他文档的关系等），默认为空字典。
        provider (Optional[str]): 提供该文档的服务或来源，默认为 "dify"。
        children (Optional[list[ChildDocument]]): 子文档列表，默认为 None。
    """

    # 主要文本内容字段，类型为字符串，不能为空
    page_content: str

    # 向量表示字段，可选，类型为浮点数列表，默认值为 None
    vector: Optional[list[float]] = None

    # 元数据字段，用于存储与文档相关的任意信息，类型为字典，默认为空字典
    metadata: dict = {}

    # 提供者字段，可选，类型为字符串，默认值为 "dify"
    provider: Optional[str] = "dify"

    # 子文档字段，可选，类型为 ChildDocument 对象的列表，默认值为 None
    children: Optional[list[ChildDocument]] = None


class BaseDocumentTransformer(ABC):
    """Abstract base class for document transformation systems.

    A document transformation system takes a sequence of Documents and returns a
    sequence of transformed Documents.

    Example:
        .. code-block:: python

            class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
                embeddings: Embeddings
                similarity_fn: Callable = cosine_similarity
                similarity_threshold: float = 0.95

                class Config:
                    arbitrary_types_allowed = True

                def transform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[Document]:
                    stateful_documents = get_stateful_documents(documents)
                    embedded_documents = _get_embeddings_from_stateful_docs(
                        self.embeddings, stateful_documents
                    )
                    included_idxs = _filter_similar_embeddings(
                        embedded_documents, self.similarity_fn, self.similarity_threshold
                    )
                    return [stateful_documents[i] for i in sorted(included_idxs)]

                async def atransform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[Document]:
                    raise NotImplementedError

    """

    @abstractmethod
    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """

    @abstractmethod
    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
