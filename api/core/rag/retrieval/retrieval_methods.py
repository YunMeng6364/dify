from enum import Enum


class RetrievalMethod(Enum):
    """
    枚举类，定义了不同的检索方法。

    枚举值:
        SEMANTIC_SEARCH: 语义搜索方法。
        FULL_TEXT_SEARCH: 全文搜索方法。
        HYBRID_SEARCH: 混合搜索方法，结合了语义搜索和全文搜索。
    """
    SEMANTIC_SEARCH = "semantic_search"
    FULL_TEXT_SEARCH = "full_text_search"
    HYBRID_SEARCH = "hybrid_search"

    @staticmethod
    def is_support_semantic_search(retrieval_method: str) -> bool:
        """
        判断给定的检索方法是否支持语义搜索。

        参数:
            retrieval_method (str): 检索方法的字符串表示。

        返回值:
            bool: 如果检索方法支持语义搜索，则返回True，否则返回False。
        """
        return retrieval_method in {RetrievalMethod.SEMANTIC_SEARCH.value, RetrievalMethod.HYBRID_SEARCH.value}

    @staticmethod
    def is_support_fulltext_search(retrieval_method: str) -> bool:
        """
        判断给定的检索方法是否支持全文搜索。

        参数:
            retrieval_method (str): 检索方法的字符串表示。

        返回值:
            bool: 如果检索方法支持全文搜索，则返回True，否则返回False。
        """
        return retrieval_method in {RetrievalMethod.FULL_TEXT_SEARCH.value, RetrievalMethod.HYBRID_SEARCH.value}
