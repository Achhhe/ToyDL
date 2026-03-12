import torch
import torch.nn as nn
import torch.nn.functional as F


def get_identity_kernel_groupwise(channels: int, groups: int) -> torch.Tensor:
    """
    创建恒等映射卷积核，支持分组卷积。
    确保每个通道组内实现恒等映射 (y = x)。

    参数:
    - channels: 总通道数
    - groups: 分组数

    返回:
    - weight: 恒等映射卷积核，形状为 (channels, channels // groups, 1, 1)
    """
    per_group = channels // groups  # 每组通道数
    weight = torch.zeros((channels, per_group, 1, 1))
    for g in range(groups):
        for i in range(per_group):
            # 将每组的对应位置设为 1，实现恒等映射
            weight[g * per_group + i, i, 0, 0] = 1.0
    return weight


def pad_1x1_to_3x3_groupwise(w: torch.Tensor) -> torch.Tensor:
    """
    将 1x1 卷积核填充为 3x3 卷积核（保持功能等价）。
    如果输入已经是 3x3 或更大，则直接返回。

    参数:
    - w: 输入卷积核权重

    返回:
    - padded_w: 填充后的卷积核
    """
    if w.shape[2] == 1 and w.shape[3] == 1:
        # 在四周填充 0，变为 3x3
        return F.pad(w, [1, 1, 1, 1])
    return w


def fuse_parallel_convs(conv1: nn.Conv2d, conv3: nn.Conv2d, channels: int, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合并行卷积：将 1x1 分组卷积、3x3 分组卷积和恒等映射融合为单一 3x3 分组卷积。

    参数:
    - conv1: 1x1 分组卷积层
    - conv3: 3x3 分组卷积层
    - channels: 通道数
    - groups: 分组数

    返回:
    - w_fused: 融合后的卷积核权重
    - b_fused: 融合后的偏置
    """
    # 获取 3x3 卷积参数
    w3, b3 = conv3.weight.data, conv3.bias.data
    
    # 将 1x1 卷积扩展为 3x3
    w1 = pad_1x1_to_3x3_groupwise(conv1.weight.data)
    b1 = conv1.bias.data
    
    # 创建恒等映射的 3x3 卷积核
    w_id = pad_1x1_to_3x3_groupwise(get_identity_kernel_groupwise(channels, groups))
    b_id = torch.zeros(channels, device=b3.device)
    
    # 融合：权重和偏置直接相加
    w_fused = w3 + w1 + w_id
    b_fused = b3 + b1 + b_id
    
    return w_fused, b_fused


def test_parallel_convs_fusion(channels: int, groups: int, input_shape: tuple) -> None:
    """
    测试并行卷积融合的等价性。

    参数:
    - channels: 通道数
    - groups: 分组数
    - input_shape: 输入张量形状 (batch, channels, height, width)
    """
    print(f"\n测试并行卷积融合 (channels={channels}, groups={groups}):")
    
    # 创建卷积层
    conv1 = nn.Conv2d(channels, channels, 1, padding=0, groups=groups, bias=True)  # 1x1 分组卷积
    conv3 = nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=True)  # 3x3 分组卷积
    x = torch.rand(*input_shape)  # 随机输入张量
    
    # 原始网络前向传播：三个分支相加 (对应残差结构)
    y_original = conv1(x) + conv3(x) + x
    
    # 融合为单一 3x3 卷积
    w_fused, b_fused = fuse_parallel_convs(conv1, conv3, channels, groups)
    conv_fused = nn.Conv2d(channels, channels, 3, padding=1, bias=True, groups=groups)
    conv_fused.weight.data = w_fused
    conv_fused.bias.data = b_fused
    y_fused = conv_fused(x)
    
    # 计算误差
    max_error = (y_original - y_fused).abs().max().item()
    print(f"并行卷积融合最大误差: {max_error:.2e}")


if __name__ == "__main__":
    # 测试配置
    channels = 8
    groups = 8  # 深度可分离卷积结构
    input_shape = (1, channels, 32, 32)
    
    test_parallel_convs_fusion(channels, groups, input_shape)