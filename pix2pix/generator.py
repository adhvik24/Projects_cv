import torch
import torch.nn as nn
import unet_block as ub


class Gen(nn.Module):
    def __init__(self, inp=3, feature_dim=64):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(inp, feature_dim, 4, 2, 1, padding_mode='reflect'),
                                nn.LeakyReLU(0.2))
        self.l2 = ub.unet_block(
            feature_dim, feature_dim*2, dir='down', act_fn='LReLU', drop=False)
        self.l3 = ub.unet_block(
            feature_dim*2, feature_dim*4, dir='down', act_fn='LReLU', drop=False)
        self.l4 = ub.unet_block(
            feature_dim*4, feature_dim*8, dir='down', act_fn='LReLU', drop=False)
        self.l5 = ub.unet_block(
            feature_dim*8, feature_dim*8, dir='down', act_fn='LReLU', drop=False)
        self.l6 = ub.unet_block(
            feature_dim*8, feature_dim*8, dir='down', act_fn='LReLU', drop=False)
        self.l7 = ub.unet_block(
            feature_dim*8, feature_dim*8, dir='down', act_fn='LReLU', drop=False)

        self.ul = nn.Sequential(nn.Conv2d(feature_dim*8, feature_dim*8, 4, 2, 1),
                                nn.ReLU()
                                )

        self.l11 = ub.unet_block(
            feature_dim*8, feature_dim*8, dir='up', act_fn='ReLU', drop=True)
        self.l12 = ub.unet_block(
            feature_dim*8*2, feature_dim*8, dir='up', act_fn='ReLU', drop=True)

        self.l13 = ub.unet_block(
            feature_dim*8*2, feature_dim*8, dir='up', act_fn='ReLU', drop=True)

        self.l14 = ub.unet_block(
            feature_dim*8*2, feature_dim*8, dir='up', act_fn='ReLU', drop=False)

        self.l15 = ub.unet_block(
            feature_dim*8*2, feature_dim*4, dir='up', act_fn='ReLU', drop=False)

        self.l16 = ub.unet_block(
            feature_dim*8, feature_dim*2, dir='up', act_fn='ReLU', drop=False)

        self.l17 = ub.unet_block(
            feature_dim*4, feature_dim, dir='up', act_fn='ReLU', drop=False)

        self.map_ = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*2, inp, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        layer1 = self.l1(x)
        layer2 = self.l2(layer1)
        layer3 = self.l3(layer2)
        layer4 = self.l4(layer3)
        layer5 = self.l5(layer4)
        layer6 = self.l6(layer5)
        layer7 = self.l7(layer6)

        b_layer = self.ul(layer7)

        layer11 = self.l11(b_layer)
        layer22 = self.l12(torch.cat([layer11, layer7], dim=1))
        layer33 = self.l13(torch.cat([layer22, layer6], dim=1))
        layer44 = self.l14(torch.cat([layer33, layer5], dim=1))
        layer55 = self.l15(torch.cat([layer44, layer4], dim=1))
        layer66 = self.l16(torch.cat([layer55, layer3], dim=1))
        layer77 = self.l17(torch.cat([layer66, layer2], dim=1))

        map_layer = self.map_(torch.cat([layer77, layer1], dim=1))

        return map_layer


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Gen(inp=3, feature_dim=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
