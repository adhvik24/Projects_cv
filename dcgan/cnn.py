import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(self, inp, features, kernel_size, stride, padding, bias=False):
        super(block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp, features, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layer(x)


class d_block(nn.Module):
    def __init__(self, inp, out, kernel_size, stride, padding):
        super(d_block, self).__init__()
        self.blayer = nn.Sequential(
            nn.ConvTranspose2d(inp, out, kernel_size,
                               stride, padding, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blayer(x)
