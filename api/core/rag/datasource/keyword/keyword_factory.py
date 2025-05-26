from typing import Any

from configs import dify_config
from core.rag.datasource.keyword.keyword_base import BaseKeyword
from core.rag.datasource.keyword.keyword_type import KeyWordType
from core.rag.models.document import Document
from models.dataset import Dataset


class Keyword:
    """
    Keyword类用于管理数据集中的关键词操作。

    Attributes:
        _dataset (Dataset): 关键词类实例初始化时绑定的数据集。
        _keyword_processor (BaseKeyword): 根据配置初始化的关键词处理器实例。
    """

    def __init__(self, dataset: Dataset):
        """
        初始化Keyword类实例。

        Args:
            dataset (Dataset): 要绑定的数据集实例。
        """
        self._dataset = dataset
        self._keyword_processor = self._init_keyword()

    def _init_keyword(self) -> BaseKeyword:
        """
        根据配置初始化关键词处理器。

        Returns:
            BaseKeyword: 初始化后的关键词处理器实例。
        """
        keyword_type = dify_config.KEYWORD_STORE
        keyword_factory = self.get_keyword_factory(keyword_type)
        return keyword_factory(self._dataset)

    @staticmethod
    def get_keyword_factory(keyword_type: str) -> type[BaseKeyword]:
        """
        根据关键词类型获取对应的工厂函数。

        Args:
            keyword_type (str): 关键词类型字符串。

        Returns:
            type[BaseKeyword]: 关键词处理器的类。

        Raises:
            ValueError: 如果关键词类型不支持则抛出异常。
        """
        # 根据关键词类型匹配并返回相应的关键词处理类
        match keyword_type:
            # 当关键词类型为JIEBA时，导入并返回Jieba类
            case KeyWordType.JIEBA:
                from core.rag.datasource.keyword.jieba.jieba import Jieba
                return Jieba
            # 当关键词类型不支持时，抛出值错误异常
            case _:
                raise ValueError(f"Keyword store {keyword_type} is not supported.")

    def create(self, texts: list[Document], **kwargs):
        """
        创建关键词。

        Args:
            texts (list[Document]): 文档列表。
            **kwargs: 其他参数。
        """
        self._keyword_processor.create(texts, **kwargs)

    def add_texts(self, texts: list[Document], **kwargs):
        """
        添加文本到关键词处理器。

        Args:
            texts (list[Document]): 文档列表。
            **kwargs: 其他参数。
        """
        self._keyword_processor.add_texts(texts, **kwargs)

    def text_exists(self, id: str) -> bool:
        """
        检查文本是否存在。

        Args:
            id (str): 文本的唯一标识符。

        Returns:
            bool: 文本是否存在。
        """
        return self._keyword_processor.text_exists(id)

    def delete_by_ids(self, ids: list[str]) -> None:
        """
        通过ID删除文本。

        Args:
            ids (list[str]): 文本的唯一标识符列表。
        """
        self._keyword_processor.delete_by_ids(ids)

    def delete(self) -> None:
        """删除所有文本。"""
        self._keyword_processor.delete()

    def search(self, query: str, **kwargs: Any) -> list[Document]:
        """
        搜索文本。

        Args:
            query (str): 查询字符串。
            **kwargs: 其他参数。

        Returns:
            list[Document]: 搜索到的文档列表。
        """
        return self._keyword_processor.search(query, **kwargs)

    def __getattr__(self, name):
        """
        动态获取属性或方法。

        Args:
            name: 属性或方法名。

        Returns:
            调用关键词处理器的同名方法。

        Raises:
            AttributeError: 如果关键词处理器没有该方法则抛出异常。
        """
        if self._keyword_processor is not None:
            method = getattr(self._keyword_processor, name)
            if callable(method):
                return method

        raise AttributeError(f"'Keyword' object has no attribute '{name}'")
