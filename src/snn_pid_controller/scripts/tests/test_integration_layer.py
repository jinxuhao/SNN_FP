import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import unittest
from layers.integration_layer import IntegrationLayer

class TestIntegrationLayer(unittest.TestCase):
    def setUp(self):
        self.integration_layer = IntegrationLayer(num_neurons=63, threshold=20)

    def test_initialization(self):
        # 测试初始化的神经元数量是否正确
        self.assertEqual(self.integration_layer.num_neurons, 63)
        self.assertEqual(self.integration_layer.threshold, 20)

    def test_integrate_positive_error(self):
        # 测试正误差信号的累积
        self.integration_layer.integrate_error(10)
        self.assertGreater(self.integration_layer.c_plus.v[0].item(), 0)  # 改为 v

    def test_integrate_negative_error(self):
        # 测试负误差信号的累积
        self.integration_layer.integrate_error(-10)
        self.assertGreater(self.integration_layer.c_minus.v[0].item(), 0)  # 改为 v

    def test_integral_state(self):
        # 检查积分状态
        self.integration_layer.integrate_error(20)  # 累积足够的正误差
        state = self.integration_layer.get_integral_state()
        self.assertIsInstance(state, int)  # 检查输出是否为整数

if __name__ == "__main__":
    unittest.main()
