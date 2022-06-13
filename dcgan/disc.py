import torch
import torch.nn as nn
import cnn


class Disc(nn.Module):
    def __init__(self, inp, features):
        super(Disc, self).__init__()
        self.discriminator = nn.Sequential(cnn.block(inp, features, 4, 2, 1, bias=True),
                                           cnn.block(
                                               features, 2*features, 4, 2, 1),
                                           cnn.block(features*2, 2 *
                                                     2*features, 4, 2, 1),
                                           cnn.block(features*2*2,
                                                     2*2*2*features, 4, 2, 1),
                                           nn.Conv2d(features*8, 1, 4, 2, 0),
                                           nn.Sigmoid()
                                           )

    def forward(self, x):
        return self.discriminator(x)
