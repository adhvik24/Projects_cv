import torch
import torch.nn as nn
import cnn
import gen as g
import disc as d


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = d.Disc(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    print(disc(x).shape)
    gen = g.Gen(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print(gen(z).shape)


if __name__ == "__main__":
    test()
