import numpy as np

# 1. 定义基础的11维归一化向量
# 对应 state_t 的11个特征
base_normal_vector = np.array([
    1e-6,  # receiving_rate (bps -> Mbps)
    1/50,  # num_received_packets
    1e-4,  # received_bytes
    1/100, # queuing_delay (ms)
    1e-12, # delay_minus_base (ms) - 使用大数进行缩放
    1e-12, # min_seen_delay (ms) - 使用大数进行缩放
    1.0,   # delay_ratio
    1/100, # delay_avg_min_diff (ms)
    1/50,  # mean_interarrival (ms)
    1/50,  # packet_jitter (ms)
    1.0,   # packet_loss_ratio
])

# 2. 将基础向量重复6次，构建完整的66维 NORMAL_VECTOR
# 对应 observation 的 (state_t-5, state_t-4, ..., state_t) 结构
NORMAL_VECTOR = np.tile(base_normal_vector, 6)

# 3. 定义您的归一化函数
def adjust_dataset(dataset_dict, b_in_Mb=1e6):
    """
    使用预定义的NORMAL_VECTOR对数据集进行归一化。
    """
    # 归一化 actions
    dataset_dict["actions"] /= b_in_Mb

    # 归一化 observations 和 next_observations
    # 使用广播机制 (element-wise multiplication)
    dataset_dict["observations"] *= NORMAL_VECTOR
    dataset_dict["next_observations"] *= NORMAL_VECTOR

    return dataset_dict

# --- 示例用法 ---
if __name__ == '__main__':
    # 假设您已经加载了 result2dataset.py 生成的数据集
    # a = np.ones((10, 66))
    # a[:, 0] = 500000
    # a[:, 4] = 1.8e12
    # dataset = {'observations': a, 'next_observations': a.copy(), 'actions': np.full((10, 1), 2000000)}
    
    # print("Original observation (first sample, first 12 features):")
    # print(dataset['observations'][0, :12])
    # print("\nOriginal action (first sample):")
    # print(dataset['actions'][0])
    
    # # 应用归一化
    # adjusted_dataset = adjust_dataset(dataset)
    
    # print("\n" + "="*30)
    # print("Normalized observation (first sample, first 12 features):")
    # print(adjusted_dataset['observations'][0, :12])
    # print("\nNormalized action (first sample):")
    # print(adjusted_dataset['actions'][0])

    print("NORMAL_VECTOR (shape: {}):".format(NORMAL_VECTOR.shape))
    print(NORMAL_VECTOR)
