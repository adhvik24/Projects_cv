import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cnn
import gen as g
import disc as d
from config import transforms as T
from utils import weights as w
import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lrd = 2e-4
lrg = 3e-4
batch_size = 128
image_size = 64
img_channels = 1
noise_d = 100
num_epochs = 5
disc_feat = 64
gen_feat = 64


data = datasets.MNIST(root='mnist/', train=True, transform=T, download=True)

dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

gen = g.Gen(noise_d, img_channels, gen_feat).to(device)
disc = d.Disc(img_channels, disc_feat).to(device)

w(gen)
w(disc)

opt_g = optim.Adam(gen.parameters(), lr=lrg, betas=(0.5, 0.999))
opt_d = optim.Adam(disc.parameters(), lr=lrd, betas=(0.5, 0.999))
criterion = nn.BCELoss()


LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64


fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_d.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_g.step()

        if batch_idx % 15 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

            step += 1
