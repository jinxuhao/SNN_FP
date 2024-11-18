import unittest
from config import Config
from network.build_network import build_snn_network

class TestSNNNetwork(unittest.TestCase):
    def setUp(self):
        # 初始化配置和构建网络
        self.config = Config()
        self.network = build_snn_network(self.config)

    def test_network_layers(self):
        # 确保网络中包含所需的层
        self.assertIn("input", self.network.layers)
        self.assertIn("counter_plus", self.network.layers)
        self.assertIn("counter_minus", self.network.layers)
        self.assertIn("shift_up", self.network.layers)
        self.assertIn("shift_down", self.network.layers)

    def test_network_run(self):
        # 测试网络是否能够正常运行
        input_data = self.network.layers["input"].generate_input()
        self.network.run(inputs={"input": input_data}, time=10)
        # 检查积分层中的电位变化（根据时间和输入数据）

if __name__ == "__main__":
    unittest.main()
