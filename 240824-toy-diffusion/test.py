import torch
from model import Denoiser
from options import *
import matplotlib.pyplot as plt
import numpy as np

model = Denoiser().to(device)
ckpt = 'latest.pth'
model.load_state_dict(torch.load(ckpt), strict=True)

n_samples = 10000
z = torch.randn(n_samples, 1).to(device)
z_ddim = z.clone()
_acc_time_by_ddim = 4
for i in range(n_steps - 1, 0, -1):
    def sample_ddpm(x, t):
        t = torch.tensor([t]).to(device)
        at = alphas.gather(-1, t).view(-1, 1)
        st = sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1)
        mean = torch.sqrt(1.0 / at) * (x - (1.0 - at) * model(x, t.repeat(n_samples)) / st)
        var = posterior_variance.gather(-1, t).view(-1, 1)
        xt_pre = mean + torch.sqrt(var) * torch.randn(n_samples, 1)
        return xt_pre

    def sample_ddim(x, t, eta = 1):
        t = torch.tensor([t]).to(device)
        at = alphas_cumprod.gather(-1, t).view(-1, 1)
        at_1 = alphas_cumprod_prev.gather(-1, t).view(-1, 1)
        sigma_t = eta * torch.sqrt((1 - at_1) / (1 - at) * (1 - at / at_1))
        xt_pre = (
            torch.sqrt(at_1 / at) * x 
            + (torch.sqrt(1 - at_1 - sigma_t ** 2) - torch.sqrt(
                (at_1 * (1 - at)) / at)) * model(x, t.repeat(n_samples))
            + sigma_t * torch.randn_like(x)
        )
        return xt_pre
    
    print(i)
    z = sample_ddpm(z, i)

    if i % _acc_time_by_ddim == 0:
        z_ddim = sample_ddim(z_ddim, i, eta=1)

plt.figure(figsize=(5,7))
plt.subplot(311)
plt.hist(data[:, 0].detach().cpu().numpy(), bins=100, range=(-1,1))
plt.title('gt')
plt.subplot(312)
plt.hist(z[:, 0].detach().cpu().numpy(), bins=100, range=(-1,1))
plt.title('ddpm')
plt.subplot(313)
plt.hist(z_ddim[:, 0].detach().cpu().numpy(), bins=100, range=(-1,1))
plt.title('ddim')
plt.tight_layout()
plt.show()
