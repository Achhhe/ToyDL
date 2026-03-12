import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将卷积层和BN层融合为一个卷积层，支持分组卷积。

    参数:
    - conv: 卷积层 (支持分组卷积)
    - bn: 对应的BN层

    返回:
    - w_fused: 融合后的卷积核权重
    - b_fused: 融合后的卷积偏置
    """
    # 获取卷积层参数
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(w.size(0), device=w.device)
    
    # 获取BN层参数
    gamma, beta = bn.weight, bn.bias
    mean, var = bn.running_mean, bn.running_var
    std = (var + bn.eps).sqrt()
    
    # 融合公式推导：
    # 原始计算流程: y = bn(conv(x)) = γ * (conv(x) - μ) / σ + β
    # 展开: y = (γ / σ) * conv(x) + (β - γ * μ / σ)
    # 由于 conv(x) = w * x + b，代入得: y = (γ / σ) * w * x + (γ / σ) * b + (β - γ * μ / σ)
    # 融合后卷积的权重: w_fused = (γ / σ) * w
    # 融合后卷积的偏置: b_fused = (γ / σ) * b + β - (γ / σ) * μ
    
    # 调整 gamma / std 的形状以匹配卷积核权重 (out_channels, 1, 1, 1)
    scale = (gamma / std).view(-1, 1, 1, 1)
    w_fused = w * scale  # 逐元素相乘
    b_fused = beta + (gamma / std) * (b - mean)  # 应用BN的缩放和平移
    
    return w_fused, b_fused


def test_fusion(conv: nn.Conv2d, bn: nn.BatchNorm2d, x: torch.Tensor, description: str) -> None:
    """
    测试卷积BN融合的等价性。

    参数:
    - conv: 卷积层
    - bn: BN层
    - x: 输入张量
    - description: 测试描述
    """
    print(f"\n测试 {description} 融合:")
    
    # 设置为评估模式
    conv.eval()
    bn.eval()
    
    # 原始模型前馈计算
    y_original = bn(conv(x))
    
    # 融合模型前馈计算
    w_fused, b_fused = fuse_conv_bn(conv, bn)
    conv_fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size, 
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation, 
        groups=conv.groups, bias=True
    )
    conv_fused.weight.data = w_fused
    conv_fused.bias.data = b_fused
    y_fused = conv_fused(x)
    
    # 计算误差
    max_error = (y_original - y_fused).abs().max().item()
    print(f"{description} BN融合最大误差: {max_error:.2e}")


if __name__ == "__main__":
    # 测试普通卷积融合
    conv = nn.Conv2d(3, 16, 3, padding=1, bias=True)
    bn = nn.BatchNorm2d(16)
    x = torch.rand(1, 3, 32, 32)  # 随机输入图像
    test_fusion(conv, bn, x, "普通卷积")
    
    # 测试分组卷积融合
    groups = 4  # 分组数
    in_channels = 16
    out_channels = 16
    conv_group = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=True)
    bn_group = nn.BatchNorm2d(out_channels)
    x_group = torch.rand(1, in_channels, 32, 32)  # 随机输入图像
    test_fusion(conv_group, bn_group, x_group, "分组卷积")