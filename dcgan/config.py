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


transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(img_channels)], [
                0.5 for _ in range(img_channels)]
        ),
    ]
)
