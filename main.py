import numpy as np
import torch
import matplotlib
import torchsummary
import torch.nn.functional as F
import datetime

from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn

idx = -1
# Ask the user for an index input
while 0 > idx or 59999 < idx:
    idx = int(input("Input an integer index between 0 and 59999\n"))


# Data processing
# Load the mnist dataset
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)



plt.imshow(train_set.data[idx], cmap='gray')
plt.show()