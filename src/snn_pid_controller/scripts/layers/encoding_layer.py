# import numpy as np
# import torch
# from bindsnet.network.nodes import Nodes

# class EncodingLayer(Nodes):
#     def __init__(self, num_neurons=63):
#         """
#         Encoding layer that uses 2D arrays to compute mathematical operations.
#         :param num_neurons: Number of neurons in the 1D input and output populations.
#         """
#         super(EncodingLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
#         self.num_neurons = num_neurons

#     def create_2D_operation_array(self, operation="add"):
#         """
#         Create a 2D operation array for addition or subtraction.
#         :param operation: Type of operation ("add" or "subtract").
#         :return: A 2D numpy array representing the operation.
#         """
#         operation_array = np.zeros((self.num_neurons, self.num_neurons), dtype=int)

#         for i in range(self.num_neurons):
#             for j in range(self.num_neurons):
#                 if operation == "add":
#                     result_index = i + j
#                 elif operation == "subtract":
#                     result_index = i - j
#                 else:
#                     raise ValueError("Unsupported operation. Use 'add' or 'subtract'.")

#                 # Map out-of-bound indices to boundary neurons
#                 result_index = max(0, min(result_index, self.num_neurons - 1))
#                 operation_array[i, j] = result_index

#         return torch.tensor(operation_array, dtype=torch.long)



#     def step(self, dt):
#         print("DEBUG___InputLayer step() called")
#         # 如果不需要自动更新状态，可以覆盖 step 方法，避免默认行为
#         pass

#     def forward(self, x):
#         """
#         Perform the forward pass and update the layer state.
#         :param x: A tensor of shape (1, num_neurons), representing input spikes.
#         """
#         print(f"Encoding___EncodingLayer received input x with shape: {x.shape}")
#         print(f"Encoding___X_tensor x: {x}")

#         # 确保输入维度正确
#         if x.dim() == 3 and x.shape[1] == 1:
#             x = x.squeeze(1)

#          # 假设 y_index 和 r_index 分别为 x 中最大值的前两个索引
#         top_indices = torch.topk(x.flatten(), 2).indices.tolist()  # 获取前两个激活最大值的索引
#         if len(top_indices) < 2:
#             print("DEBUG___Warning: Less than two inputs active for encoding.")
#             self.s = torch.zeros(1, self.num_neurons)  # 重置状态
#             return self.s

#         y_index, r_index = top_indices
#         # print(f"DEBUG___y_index: {y_index}, r_index: {r_index}")

#         # 验证是否与 InputLayer 的生成逻辑一致
#         print(f"CHECK___Expected y_index from InputLayer: {y_index}")
#         print(f"CHECK___Expected r_index from InputLayer: {r_index}")
#         return self.s
import numpy as np
import torch
from bindsnet.network.nodes import Nodes

class EncodingLayer(Nodes):
    def __init__(self, num_neurons=63, e_t_range=(-20, 20)):
        """
        Encoding layer that uses 2D arrays to compute mathematical operations.
        :param num_neurons: Number of neurons in the 1D input and output populations.
        :param e_t_range: The range of e_t values (min, max).
        """
        super(EncodingLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.e_t_range = e_t_range  # (min, max)
        self.operation_array = self.create_2D_operation_array()

        # 用于控制是否使用显式索引
        self.use_indices = False
        self.y_index = None
        self.r_index = None


    def create_2D_operation_array(self):
        """
        Create a 2D operation array for subtraction.
        Each diagonal corresponds to a specific e_t value.
        """
        operation_array = np.zeros((self.num_neurons, self.num_neurons), dtype=int)
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                # Calculate the diagonal index
                diagonal_index = j - i
                operation_array[i, j] = diagonal_index
        return operation_array

    def map_e_t_to_neuron(self, e_t):
        """
        Map e_t value to the corresponding neuron index in the 1D output array.
        """
        e_t_min, e_t_max = self.e_t_range
        e_t_clipped = max(e_t_min, min(e_t, e_t_max))  # Clip e_t to range
        # Map e_t to the neuron index
        neuron_index = int(round((e_t_clipped - e_t_min) / (e_t_max - e_t_min) * (self.num_neurons - 1)))
        return neuron_index

    def forward(self, x):
        """
        Perform the forward pass and update the layer state.
        :param x: A tensor of shape (1, num_neurons), representing input spikes.
        """
        x = x.float()
        # print(f"Encoding___EncodingLayer received input x with shape: {x.shape}")
        # print(f"Encoding___X_tensor x: {x}")

        # Ensure x is in the correct shape
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # 使用显式传递的索引
        if self.use_indices and self.y_index is not None and self.r_index is not None:
            y_index = self.y_index
            r_index = self.r_index
            print(f"Encoding___y_index_1: {y_index}, r_index: {r_index}")
        else:

            # Get active indices
            active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
            if len(active_indices) < 2:
                print("Encoding___Warning: Less than two inputs active for encoding.")
                self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)  # Reset state
                return self.s

            # Extract y_index and r_index
            y_index, r_index = active_indices[:2]
            print(f"Encoding___y_index: {y_index}, r_index: {r_index}")

        # Calculate the diagonal index
        diagonal_index = self.operation_array[y_index, r_index]
        # print(f"Encoding___diagonal_index: {diagonal_index}")
        # Map diagonal index to e_t value
        e_t_value = ((self.num_neurons-1) + diagonal_index) /(2 * (self.num_neurons - 1)) * (self.e_t_range[1] - self.e_t_range[0])
        # e_t_value = (( diagonal_index) /(self.num_neurons - 1)) * (self.e_t_range[1] - self.e_t_range[0])
        e_t_value += self.e_t_range[0]
        print(f"Encoding___e_t_value: {e_t_value}")

        # Map e_t value to neuron index
        result_index = self.map_e_t_to_neuron(e_t_value)
        print(f"Encoding___result_index: {result_index}")

        # Update the state
        self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)
        self.s[0, result_index] = 1
        # print(f"Encoding___self.s: {self.s}")
        # print(f"Encoding___Output state s shape: {self.s.shape}")
        # print(f"Encoding___Output of s (source spikes): {self.s.shape}, Data type: {self.s.dtype}")
        # # self.s = (self.s > 0).bool()
        x = x.view(x.size(0), -1)
        return self.s
