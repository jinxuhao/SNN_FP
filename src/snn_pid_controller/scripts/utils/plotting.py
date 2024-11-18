import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import matplotlib.pyplot as plt

def plot_layer(ax, value, title, min_val, max_val, num_neurons):
    """绘制神经元激活层的函数"""
    idx = int((value - min_val) / (max_val - min_val) * (num_neurons - 1))
    spikes = np.zeros(num_neurons)
    spikes[idx] = 1

    ax.clear()
    ax.scatter(range(num_neurons), spikes, color="blue")
    ax.set_title(title)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Neuron Index")
