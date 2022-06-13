import torch
import torch.nn as nn


class unet_block(nn.Module):
    def __init__(self, inp, out, dir='down', act_fn="ReLU", drop=False):
        super(unet_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp, out, 4, 2, 1, bias=False, padding_mode='reflect') if dir == 'down'
            else
            nn.ConvTranspose2d(inp, out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU() if act_fn == 'ReLU' else nn.LeakyReLU(0.2),
        )
        self.drop = drop
        self.dropout = nn.Dropout(0.5)
        self.dir = dir

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x)
