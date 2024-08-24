from torch.utils.data import DataLoader, TensorDataset
from model import Denoiser
from time import time
from options import *

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)

# train
denoiser = Denoiser().to(device)
optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.01)
denoiser.train()

tic = time()
for e in range(n_epoches):
    tic_loc = time()
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        x = data[0].to(device).float()
        b = x.shape[0]

        # sample t and noise
        t = torch.randint(0, n_steps, (b,)).to(device)
        z = torch.randn(b, 1).to(device)

        # forward sample xt
        xt = sqrt_alphas_cumprod.gather(-1, t).view(-1, 1) * x + sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1) * z

        # denoise
        z_pred = denoiser(xt, t)

        loss = F.mse_loss(z, z_pred)
        print(f'epoch={e}, iter={i}/{len(dataloader)}, loss={loss.item():.4f}')
        loss.backward()
        optimizer.step()
    toc = time() - tic_loc
    print(f'epch={e} takes {toc:.2f}s')

torch.save(denoiser.state_dict(), 'latest.pth')
toc = time() - tic
print(f'totally takes {toc:.2f}s')