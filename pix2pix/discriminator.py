import torch
import torch.nn as nn
import cnn


class Disc(nn.Module):
    def __init__(self, inp=3, feature_dim=[64, 128, 256, 512]):
        super(Disc, self).__init__()
        convlayers = []
        input = feature_dim[0]
        convlayers.append(cnn.CNN(inp*2, input, 4, 2),)
        for feature in feature_dim[1:]:
            convlayers.append(
                cnn.CNNB(input, feature, 4, 1 if feature == feature_dim[-1] else 2))
            input = feature
        convlayers.append(nn.Conv2d(input, 1, kernel_size=4,
                          stride=1, padding=1, padding_mode='reflect'),)
        self.model = nn.Sequential(*convlayers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)
