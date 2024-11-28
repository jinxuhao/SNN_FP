import torch

def create_weight_matrix(n_input, n_output):
    """
    创建一个稀疏块对角矩阵，将多个输入映射到指定的输出范围。
    
    :param n_input: 每个输入层的神经元数量。
    :param n_output: 输出层的总神经元数量，必须是 n_input 的整数倍。
    :return: 权重矩阵，形状为 (n_input, n_output)。
    """
    if n_output % n_input != 0:
        raise ValueError("n_output 必须是 n_input 的整数倍！")

    # 创建全零权重矩阵
    w = torch.zeros(n_input, n_output)

    # 计算块的数量
    num_blocks = n_output // n_input

    # 为每个块分配单位矩阵
    for i in range(num_blocks):
        start_idx = i * n_input
        end_idx = start_idx + n_input
        w[:, start_idx:end_idx] = torch.eye(n_input)

    print(f"Shape of create_weight_matrix (connection weights): {w.shape}")

    return w
