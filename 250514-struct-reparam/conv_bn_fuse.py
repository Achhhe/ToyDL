import torch
import torch.nn as nn
import torch.nn.functional as F

def fuse_conv_bn(conv, bn):
    """
    将卷积层和BN层融合为一个卷积层
    
    参数:
    - conv: 卷积层
    - bn: 对应的BN层
    
    返回:
    - w_fused: 融合后的卷积核权重
    - b_fused: 融合后的卷积偏置
    """
    # 获取卷积层参数
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(w.size(0)).to(w.device)
    
    # 获取BN层参数
    gamma, beta = bn.weight, bn.bias
    mean, var = bn.running_mean, bn.running_var
    std = (var + bn.eps).sqrt()
    
    # 融合公式推导：
    # 原始计算流程: y = bn(conv(x)) = γ * (conv(x)-μ)/σ + β
    # 展开: y = (γ/σ) * conv(x) + (β - γ*μ/σ)
    # 由于conv(x)=w*x+b，代入得: y = (γ/σ)*w*x + (γ/σ)*b + (β - γ*μ/σ)
    # 融合后卷积的权重: w_fused = (γ/σ)*w
    # 融合后卷积的偏置: b_fused = (γ/σ)*b + β - (γ/σ)*μ
    
    w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)  # 调整通道维度以匹配卷积核形状
    b_fused = beta + (gamma / std) * (b - mean)  # 应用BN的缩放和平移
    
    return w_fused, b_fused
  
# 验证融合前后等价
conv = nn.Conv2d(3, 16, 3, padding=1, bias=True)
bn = nn.BatchNorm2d(16)
conv.eval()  # 设置为评估模式，确保BN使用全局统计量
bn.eval()    # 同上
x = torch.rand(1, 3, 32, 32)  # 随机输入图像

# 原始模型前馈计算
y = bn(conv(x))

# 融合模型前馈计算
w_fused, b_fused = fuse_conv_bn(conv, bn)
conv_fused = nn.Conv2d(3, 16, 3, padding=1, bias=True)
conv_fused.weight.data, conv_fused.bias.data = w_fused, b_fused  # 加载融合参数
y_fused = conv_fused(x)  # 直接通过融合后的卷积层计算

# 输出误差：理想情况下应接近浮点数精度(1e-7级别)
print('卷积BN融合最大误差：', (y-y_fused).abs().max().item())