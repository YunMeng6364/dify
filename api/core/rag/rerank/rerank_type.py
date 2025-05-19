from enum import StrEnum


class RerankMode(StrEnum):
    """
    RerankMode 是一个字符串枚举类，用于定义重排序的模式。

    该类继承自 StrEnum，表示枚举值是字符串类型。RerankMode 提供了两种重排序模式：
    1. RERANKING_MODEL: 使用重排序模型进行重排序。
    2. WEIGHTED_SCORE: 使用加权分数进行重排序。

    枚举值:
    - RERANKING_MODEL: 表示使用重排序模型进行重排序的模式。
    - WEIGHTED_SCORE: 表示使用加权分数进行重排序的模式。
    """
    RERANKING_MODEL = "reranking_model"
    WEIGHTED_SCORE = "weighted_score"
