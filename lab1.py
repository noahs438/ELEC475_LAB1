import argparse
import torch
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from model import autoencoderMLP4Layer
import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, required=True)
    arguments = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)

    # Data processing
    # Loading in the mnist dataset for TESTING
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, 1, shuffle=False)

    # Import the model
    autoencoder = autoencoderMLP4Layer(N_bottleneck=8).to(device)
    autoencoder.load_state_dict(torch.load(arguments.l))

    # Make a list of random images to use for testing
    images = []
    for _ in range(3):
        idx = random.sample(range(len(test_set)), 2)
        imgs = [test_set[i][0] for i in idx]
        images.append(imgs)


    # Testing
    print('=======================================')
    print('Testing ')
    print('=======================================')

    # print('Step 4')
    # train.test(autoencoder, test_loader, device)

    # print('Step 5')
    # train.noise_test(autoencoder, images, device)

    print('Step 6')
    train.interpolate(autoencoder, images, 8, device)



