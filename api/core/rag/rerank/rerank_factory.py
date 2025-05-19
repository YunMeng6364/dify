from core.rag.rerank.rerank_base import BaseRerankRunner
from core.rag.rerank.rerank_model import RerankModelRunner
from core.rag.rerank.rerank_type import RerankMode
from core.rag.rerank.weight_rerank import WeightRerankRunner


class RerankRunnerFactory:
    """
    RerankRunnerFactory 的工厂类，通过静态方法 create_rerank_runner 动态创建不同类型的重排序运行器实例
    """
    @staticmethod
    def create_rerank_runner(runner_type: str, *args, **kwargs) -> BaseRerankRunner:
        match runner_type:
            case RerankMode.RERANKING_MODEL.value:
                return RerankModelRunner(*args, **kwargs)
            case RerankMode.WEIGHTED_SCORE.value:
                return WeightRerankRunner(*args, **kwargs)
            case _:
                raise ValueError(f"Unknown runner type: {runner_type}")
