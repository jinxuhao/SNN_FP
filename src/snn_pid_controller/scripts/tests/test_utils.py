import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import plot_layer

class TestPlottingUtils(unittest.TestCase):
    def setUp(self):
        # 设置测试数据
        self.num_neurons = 63
        self.value = 20
        self.min_val = -40
        self.max_val = 40

    def test_plot_layer(self):
        # 测试 plot_layer 函数能否正常运行而不报错
        fig, ax = plt.subplots()
        try:
            plot_layer(ax, self.value, "Test Layer", self.min_val, self.max_val, self.num_neurons)
        except Exception as e:
            self.fail(f"plot_layer raised an exception: {e}")
        
        # 断言绘图标题是否正确
        self.assertEqual(ax.get_title(), "Test Layer")
        
        # 断言 X 轴标签是否正确
        self.assertEqual(ax.get_xlabel(), "Neuron Index")

        plt.close(fig)  # 关闭图形，避免测试时生成多余窗口

if __name__ == "__main__":
    unittest.main()
