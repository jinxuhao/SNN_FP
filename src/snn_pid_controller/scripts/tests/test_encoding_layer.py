import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
from layers.encoding_layer import EncodingLayer

class TestEncodingLayer(unittest.TestCase):
    def setUp(self):
        self.encoding_layer = EncodingLayer(num_neurons=63)

    def test_create_2D_operation_array_add(self):
        # 测试加法操作数组
        operation_array = self.encoding_layer.create_2D_operation_array(operation="add")
        self.assertEqual(operation_array[0, 0], 1)
        self.assertEqual(operation_array[1, 1], 1)
        self.assertEqual(operation_array[2, 2], 1)

    def test_perform_operation_addition(self):
        # 测试加法运算
        result = self.encoding_layer.perform_operation(0.5, 0.3, operation="add")
        self.assertIsInstance(result, int)

    def test_perform_operation_subtraction(self):
        # 测试减法运算
        result = self.encoding_layer.perform_operation(0.8, 0.2, operation="subtract")
        self.assertIsInstance(result, int)

if __name__ == "__main__":
    unittest.main()
