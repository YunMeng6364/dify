from controllers.console.datasets.hit_testing_base import DatasetsHitTestingBase
from controllers.service_api import api
from controllers.service_api.wraps import DatasetApiResource


class HitTestingApi(DatasetApiResource, DatasetsHitTestingBase):
    """
    HitTestingApi 类用于处理数据集命中测试的API请求。
    该类继承自 DatasetApiResource 和 DatasetsHitTestingBase，提供了对数据集进行命中测试的功能。
    """

    def post(self, tenant_id, dataset_id):
        """
        处理POST请求，执行数据集命中测试。

        Args:
            tenant_id (str): 租户ID，标识请求所属的租户。
            dataset_id (uuid): 数据集ID，标识要进行命中测试的数据集。

        Returns:
            dict: 命中测试的结果，通常包含命中数据或相关统计信息。
        """
        # 将数据集ID转换为字符串格式
        dataset_id_str = str(dataset_id)

        # 获取并验证数据集是否存在及有效
        dataset = self.get_and_validate_dataset(dataset_id_str)

        # 解析请求参数
        args = self.parse_args()

        # 检查命中测试参数的有效性
        self.hit_testing_args_check(args)

        # 执行命中测试并返回结果
        return self.perform_hit_testing(dataset, args)


# 将 HitTestingApi 类注册到API路由中，支持两种URL路径
api.add_resource(HitTestingApi, "/datasets/<uuid:dataset_id>/hit-testing", "/datasets/<uuid:dataset_id>/retrieve")
