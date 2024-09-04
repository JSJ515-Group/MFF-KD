import numpy as np

def compute_channel_attention_map(A):
    # 计算通道注意力图 Gc
    Gc = np.sum(np.abs(A), axis=(1, 2)) / (A.shape[1] * A.shape[2])  # 在高度和宽度维度上计算绝对值的均值
    return Gc

def compute_spatial_attention_map(A):
    # 计算空间注意力图 Gs
    Gs = np.sum(np.abs(A), axis=0) / A.shape[0]  # 在通道维度上计算绝对值的均值
    return Gs


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def compute_attention_masks(AS, AT, T, H, W, C):
    # 计算 Gs 和 Gc
    Gs_S = compute_spatial_attention_map(AS)
    Gs_T = compute_spatial_attention_map(AT)

    Gc_S = compute_channel_attention_map(AS)
    Gc_T = compute_channel_attention_map(AT)

    # 根据新公式计算 Ms 和 Mc
    Ms = H * W * softmax((Gs_S + Gs_T) / T)
    Mc = C * softmax((Gc_S + Gc_T) / T)

    return Ms, Mc


# 示例参数
C, H, W = 3, 4, 5
AS = np.random.rand(C, H, W)  # 学生的特征图
AT = np.random.rand(C, H, W)  # 老师的特征图
T = 1.0  # 温度参数

Ms, Mc = compute_attention_masks(AS, AT, T, H, W, C)

print("空间注意力掩码 Ms 的形状:", Ms.shape)
print("通道注意力掩码 Mc 的形状:", Mc.shape)
# def softmax(x, axis=None):
#     e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
#     return e_x / np.sum(e_x, axis=axis, keepdims=True)
#
#
# def compute_attention_masks(AS, AT, T, H, W, C):
#     # 计算 Gs 和 Gc
#     Gs_S = compute_spatial_attention_map(AS)
#     Gs_T = compute_spatial_attention_map(AT)
#
#     Gc_S = compute_channel_attention_map(AS)
#     Gc_T = compute_channel_attention_map(AT)
#
#     # 使用softmax计算Ms和Mc
#     Ms_softmax = softmax((Gs_S + Gs_T) / T, axis=None)  # 对整个数组应用softmax
#     Mc_softmax = softmax((Gc_S + Gc_T) / T, axis=None)  # 对整个数组应用softmax
#
#     # 根据新形状要求调整Ms和Mc
#     # 注意这里不再需要乘以H*W或C，因为形状调整已经考虑到了这些维度
#     Ms = np.reshape(Ms_softmax, (1, H, W))
#     Mc = np.reshape(Mc_softmax, (C, 1, 1))
#
#     return Ms, Mc
#
#
# # 示例参数
# C, H, W = 3, 4, 5
# AS = np.random.rand(C, H, W)  # 学生的特征图
# AT = np.random.rand(C, H, W)  # 老师的特征图
# T = 1.0  # 温度参数
#
# Ms, Mc = compute_attention_masks(AS, AT, T, H, W, C)
#
# print("空间注意力掩码 Ms 的形状:", Ms.shape)
# print("通道注意力掩码 Mc 的形状:", Mc.shape)
