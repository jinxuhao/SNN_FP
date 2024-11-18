import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
from layers.input_layer import InputLayer

class TestInputLayer(unittest.TestCase):
    def setUp(self):
        self.num_neurons = 63
        self.angle_range = (-40, 40)
        self.setpoint_range = (-25, 25)
        self.angular_velocity_range = (-80, 80)
        self.input_layer = InputLayer(self.num_neurons, self.angle_range, self.setpoint_range, self.angular_velocity_range)

    def test_initialization(self):
        self.assertEqual(self.input_layer.num_neurons, self.num_neurons)
        self.assertEqual(self.input_layer.angle_range, self.angle_range)
        self.assertEqual(self.input_layer.setpoint_range, self.setpoint_range)
        self.assertEqual(self.input_layer.angular_velocity_range, self.angular_velocity_range)

    def test_generate_virtual_input(self):
        current_angle, target_angle = self.input_layer.generate_virtual_input()
        
        self.assertGreaterEqual(current_angle, self.angle_range[0])
        self.assertLessEqual(current_angle, self.angle_range[1])
        
        self.assertGreaterEqual(target_angle, self.setpoint_range[0])
        self.assertLessEqual(target_angle, self.setpoint_range[1])

    def test_encode_index(self):
        encoded_index = self.input_layer.encode_index(0, self.angle_range[0], self.angle_range[1])
        self.assertGreaterEqual(encoded_index, 0)
        self.assertLess(encoded_index, self.num_neurons)

        encoded_min = self.input_layer.encode_index(self.angle_range[0], self.angle_range[0], self.angle_range[1])
        encoded_max = self.input_layer.encode_index(self.angle_range[1], self.angle_range[0], self.angle_range[1])
        self.assertEqual(encoded_min, 0)
        self.assertEqual(encoded_max, self.num_neurons - 1)

if __name__ == "__main__":
    unittest.main()
