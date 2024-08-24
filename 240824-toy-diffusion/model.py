import torch
import torch.nn as nn

class Denoiser(nn.Module):
    def __init__(self, dim_in=2):
        super(Denoiser, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, t):
        '''input:
                x:(b, 1)
                t:(b, )
            output:
                (b, 1)
        '''
        x = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.mlp(x)