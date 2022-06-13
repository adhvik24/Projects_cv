import torch
import torch.nn as nn
import cnn


class Gen(nn.Module):
    def __init__(self, inp, img, features):
        super(Gen, self).__init__()
        self.generator = nn.Sequential(
            cnn.d_block(inp, features*16, 4, 1, 0),
            cnn.d_block(features*16, features*8, 4, 2, 1),
            cnn.d_block(features*8, features*4, 4, 2, 1),
            cnn.d_block(features*4, features*2, 4, 2, 1),
            nn.ConvTranspose2d(features*2, img, 4, 2, 1),
            nn.Tanh()

        )

    def forward(self, x):
        return self.generator(x)
