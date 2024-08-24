import torch
import torch.nn.functional as F

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## hyper-parameters
# diffusion settings
n_steps = 1000
start, end = 1e-6, 1e-2  # 要比较小，否则n_steps太大会导致alphas_cumprod累乘为0
betas = torch.linspace(start, end, n_steps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod ** 2)

alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]])
posterior_variance = (1. - alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# train settings
n_epoches = 100
batchsize = 64
num_workers = 0

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

print(torch.randn(5, 2))

# data settings
data = torch.cat([
    torch.randn(30000, 1).to(device) * 0.01 + 0.5,
    torch.randn(40000, 1).to(device) * 0.03 + 0.1,
    torch.randn(30000, 1).to(device) * 0.01 - 0.4,
])