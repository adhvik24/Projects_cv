import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, inp, out, stride=2, kernel_size=4):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(inp, out, stride, kernel_size,
                      bias=False, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        return self.conv_layer(x)


class CNNB(nn.Module):
    def __init__(self, inp, out, stride=2, kernel_size=4):
        super(CNNB, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(inp, out, stride, kernel_size, padding=1,
                      bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        return self.conv_layer(x)
