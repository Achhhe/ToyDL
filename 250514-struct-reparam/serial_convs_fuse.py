import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_serial_3layer_groups(
    w1: torch.Tensor, b1: torch.Tensor, 
    w2: torch.Tensor, b2: torch.Tensor, 
    w3: torch.Tensor, b3: torch.Tensor, 
    groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合三个串行卷积层为一个等效卷积层，支持分组卷积。
    
    参数:
    - w1, b1: 第一个卷积层的权重和偏置
    - w2, b2: 第二个卷积层的权重和偏置
    - w3, b3: 第三个卷积层的权重和偏置
    - groups: 分组卷积的组数
    
    返回:
    - w_fused: 融合后的卷积核权重
    - b_fused: 融合后的偏置
    """
    # 计算每组通道数
    in_per_group = w1.shape[0] // groups
    mid_per_group = w2.shape[0] // groups
    out_per_group = w3.shape[0] // groups
    
    ws, bs = [], []
    # 对每个通道组分别进行融合
    for g in range(groups):
        # 获取当前组的权重和偏置切片
        w1g = w1[g * in_per_group: (g + 1) * in_per_group]
        b1g = b1[g * in_per_group: (g + 1) * in_per_group] if b1 is not None else None
        w2g = w2[g * mid_per_group: (g + 1) * mid_per_group]
        b2g = b2[g * mid_per_group: (g + 1) * mid_per_group] if b2 is not None else None
        w3g = w3[g * out_per_group: (g + 1) * out_per_group]
        b3g = b3[g * out_per_group: (g + 1) * out_per_group] if b3 is not None else None
        
        # 第一步：融合前两个卷积层 conv1 -> conv2
        # 使用卷积操作实现权重矩阵乘法，w1(1x1)作为卷积核避免padding
        w12 = F.conv2d(w2g, w1g.permute(1, 0, 2, 3), bias=None)
        # 计算融合后的偏置
        b12 = (
            (w2g * b1g.reshape(1, -1, 1, 1)).sum((1, 2, 3)) if b1g is not None else 0
        ) + (b2g if b2g is not None else 0)
        
        # 第二步：融合前两个卷积的结果与第三个卷积 conv12 -> conv3
        # 使用卷积操作实现权重矩阵乘法，w3(1x1)作为卷积核避免padding
        # 注意需要处理卷积的翻转和维度排列
        w123 = F.conv2d(
            w12.flip(2, 3).permute(1, 0, 2, 3), w3g, 
            padding=0, stride=1
        ).flip(2, 3).permute(1, 0, 2, 3)
        # 计算最终偏置
        b123 = (
            (w3g * b12.reshape(1, -1, 1, 1)).sum((1, 2, 3)) 
            if isinstance(b12, torch.Tensor) else 0
        ) + (b3g if b3g is not None else 0)
        
        ws.append(w123)
        bs.append(b123)
    
    # 拼接所有组的结果
    return torch.cat(ws, dim=0), torch.cat(bs, dim=0)


def test_serial_convs_fusion(
    channels: int, groups: int, gain: int, input_shape: tuple
) -> None:
    """
    测试串行卷积融合的等价性。
    
    参数:
    - channels: 输入通道数
    - groups: 分组数
    - gain: 通道扩展倍数
    - input_shape: 输入张量形状 (batch, channels, height, width)
    """
    print(f"\n测试串行卷积融合 (channels={channels}, groups={groups}, gain={gain}):")
    
    # 创建三个串行卷积层：1x1降维 -> 3x3特征提取 -> 1x1升维
    conv1 = nn.Conv2d(channels, gain * channels, 1, padding=0, groups=groups, bias=True)
    conv2 = nn.Conv2d(gain * channels, gain * channels, 3, padding=1, groups=groups, bias=True)  # 高维提取特征
    conv3 = nn.Conv2d(gain * channels, channels, 1, padding=0, groups=groups, bias=True)
    x = torch.rand(*input_shape)  # 随机输入张量
    
    # 原始网络前向传播
    y_original = conv3(conv2(conv1(x)))
    
    # 融合为单一3x3卷积
    w1, b1 = conv1.weight.data, conv1.bias.data
    w2, b2 = conv2.weight.data, conv2.bias.data
    w3, b3 = conv3.weight.data, conv3.bias.data
    w_fused, b_fused = fuse_serial_3layer_groups(w1, b1, w2, b2, w3, b3, groups)
    conv_fused = nn.Conv2d(channels, channels, 3, padding=1, bias=True, groups=groups)
    conv_fused.weight.data = w_fused
    conv_fused.bias.data = b_fused  # 加载融合参数
    y_fused = conv_fused(x)  # 直接通过融合后的卷积层计算
    
    # 验证融合前后输出的误差（排除边界区域，因为padding可能导致微小差异）
    max_error = (y_original - y_fused).abs()[:, :, 4:-4, 4:-4].max().item()
    print(f"串行卷积融合最大误差: {max_error:.2e}")


if __name__ == "__main__":
    # 测试配置
    channels = 8  # 输入通道数
    groups = 1    # 分组数
    gain = 3      # 通道扩展倍数
    input_shape = (1, channels, 32, 32)
    
    test_serial_convs_fusion(channels, groups, gain, input_shape)