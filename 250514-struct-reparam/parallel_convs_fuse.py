import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建恒等映射卷积核（用于残差连接）
# 支持分组卷积，确保每个通道组内实现恒等映射
def get_identity_kernel_groupwise(channels, groups):
    per_group = channels // groups  # 每组通道数
    # 创建全零权重，形状为[输出通道, 输入通道/组, 1, 1]
    weight = torch.zeros((channels, per_group, 1, 1))
    for g in range(groups):
        for i in range(per_group):
            # 将每组的对应位置设为1，实现恒等映射 y=x
            weight[g*per_group+i, i, 0, 0] = 1.0
    return weight

# 将1x1卷积核填充为3x3卷积核（保持功能等价）
def pad_1x1_to_3x3_groupwise(w):
    # 如果是1x1卷积核，则在四周填充0变为3x3
    return F.pad(w, [1,1,1,1]) if w.shape[2] == 1 and w.shape[3] == 1 else w

# 网络配置
C, groups = 8, 8   # 通道数和分组数（这里使用深度可分离卷积结构）
conv1 = nn.Conv2d(C, C, 1, padding=0, groups=groups, bias=True)  # 1x1分组卷积
conv3 = nn.Conv2d(C, C, 3, padding=1, groups=groups, bias=True)  # 3x3分组卷积
x = torch.rand(1, C, 32, 32)  # 随机输入张量

# 原始网络前向传播：三个分支相加
y = conv1(x) + conv3(x) + x  # 对应残差结构中的三个分支

# 融合为单一3x3卷积
w3, b3 = conv3.weight.data, conv3.bias.data  # 获取3x3卷积参数
w1, b1 = pad_1x1_to_3x3_groupwise(conv1.weight.data), conv1.bias.data  # 将1x1卷积扩展为3x3
w_id = pad_1x1_to_3x3_groupwise(get_identity_kernel_groupwise(C, groups))  # 创建恒等映射的3x3卷积核
b_id = torch.zeros(C)  # 恒等映射的偏置为0

# 关键融合步骤：权重和偏置直接相加
w_fused = w3 + w1 + w_id  # 合并三个卷积核的权重
b_fused = b3 + b1 + b_id  # 合并偏置项

# 创建融合后的卷积层
conv_fused = nn.Conv2d(C, C, 3, padding=1, bias=True, groups=groups)
conv_fused.weight.data, conv_fused.bias.data = w_fused, b_fused  # 加载融合参数
y_fused = conv_fused(x)  # 直接通过融合后的卷积层计算

# 验证融合前后输出的误差，理想情况下应接近浮点数精度(1e-7级别)
print('并行卷积融合最大误差：', (y-y_fused).abs().max().item())