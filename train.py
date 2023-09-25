import argparse
import datetime
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import autoencoderMLP4Layer
import matplotlib.pyplot as plt
from torchsummary import summary


# COPIED FROM THE LECTURE SLIDES (as directed to in the lab)
def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, plotPath, savePath):
    print('training...')
    model.train()  # keep track of gradient for backtracking
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs, _ in train_loader:  # imgs is a minibatch of data

            imgs = imgs.view(imgs.size(0), -1).to(device)  # flatten the image & move to device

            outputs = model(imgs)  # forward propagation through network
            loss = loss_fn(outputs, imgs)  # calculate loss
            optimizer.zero_grad()  # reset optimizer gradients to zero
            loss.backward()  # calculate loss gradients
            optimizer.step()  # iterate the optimization,
            loss_train += loss.item()

        scheduler.step(loss_train)  # update optimization hyperparameters

        losses_train += [loss_train / len(train_loader)]  # update the value of losses

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)))

    # Plotting the training loss
    plt.plot(losses_train)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(plotPath)
    plt.close()

    # Save the model
    torch.save(model.state_dict(), savePath)

    return losses_train


# Testing function for the model
def test(model, test_loader, device, defaultImageCount=3):  # Will run 3 tests by default
    model.eval()  # set the model to testing mode
    images = []

    with torch.no_grad():  # disable gradient calculation because we are in testing mode
        for idx, (imgs, label) in enumerate(test_loader):
            imgs = imgs.view(imgs.size(0), -1).to(device)  # flatten the image & move to device

            outputs = model(imgs)  # forward propagation through network

            output = outputs.view(1, 28, 28).cpu().numpy()  # reshape the output to a 28x28 image
            original = imgs.view(1, 28, 28).cpu().numpy()  # reshape the input to a 28x28 image

            # Append the images to the list
            images.append((original[0], output[0]))

        plt.figure(figsize=(9, 5))
        for idx, (original, output) in enumerate(images):
            if idx >= defaultImageCount:
                # Break if we have reached the set image count
                break

            # plot the original image
            plt.subplot(1, 2, 1)
            plt.imshow(original, cmap='gray')
            plt.title('Original')
            # plot the reconstructed image
            plt.subplot(1, 2, 2)
            plt.imshow(output, cmap='gray')
            plt.title('Reconstructed')
            plt.show()
            # plt.savefig('Part4Test.png')


def add_gaussian_noise(input):
    noise = torch.randn(input.size()) * 0.2  # 20% noise - factoring down to reduce how noisy the image is
    return input + noise


def noise_test(model, test_loader, device, defaultImageCount=3):  # Will run 3 tests by default
    model.eval()  # set the model to testing mode
    images = []

    with torch.no_grad():  # disable gradient calculation because we are in testing mode
        for idx, (imgs, label) in enumerate(test_loader):
            imgs = imgs.view(imgs.size(0), -1).to(device)  # flatten the image & move to device

            noisy_images = add_gaussian_noise(imgs)
            outputs = model(noisy_images)  # forward propagation through network

            output = outputs.view(1, 28, 28).cpu().numpy()  # reshape the output to a 28x28 image
            noisy_image = noisy_images.view(1, 28, 28).cpu().numpy()  # reshape the noisy image to a 28x28 image
            original = imgs.view(1, 28, 28).cpu().numpy()  # reshape the input to a 28x28 image

            # Append the images to the list
            images.append((original[0], noisy_image[0], output[0]))

        plt.figure(figsize=(9, 9))
        for idx, (original, noisy, output) in enumerate(images):
            if idx >= defaultImageCount:
                # Break if we have reached the set image count
                break

            # plot the original image
            plt.subplot(1, 3, 1)
            plt.imshow(original, cmap='gray')
            plt.title('Original')
            # plot the noisy image
            plt.subplot(1, 3, 2)
            plt.imshow(noisy, cmap='gray')
            plt.title('Noisy')
            # plot the reconstructed image
            plt.subplot(1, 3, 3)
            plt.imshow(output, cmap='gray')
            plt.title('Reconstructed')
            plt.show()
            # plt.savefig('Part5Test.png')


# Linear interpolation function
def interpolate(model, test_loader, steps, device, defaultImageCount=3):
    model.eval()

    with torch.no_grad():
        for idx, (img1, img2) in enumerate(test_loader):
            # Pass 2 images through the encode method
            tensor_img1 = model.encode(img1.view(1, -1).to(device))
            tensor_img2 = model.encode(img2.view(1, -1).to(device))

            tensors = []
            # Interpolate between the 2 images
            for i in torch.linspace(0, 1, steps, device=device):
                interp_tensor = i * tensor_img1 + (1 - i) * tensor_img2
                # Decode the interpolated tensor
                output = model.decode(interp_tensor.unsqueeze(0)).view(1, 28, 28)
                tensors.append(output)

            # Plot the images
            plt.figure(figsize=(16, len(test_loader)))
            plt.subplot(defaultImageCount, steps+2, (steps+2)*idx+1)
            plt.imshow(img2.squeeze().cpu().numpy(), cmap='gray')

            # Plot the interpolated images
            for k, _ in enumerate(tensors):
                plt.subplot(defaultImageCount, steps+2, (steps+2)*idx+k+2)
                plt.imshow(tensors[k].squeeze().cpu().numpy(), cmap='gray')

            # Plot the second image
            plt.subplot(defaultImageCount, steps+2, (steps+2)*idx+steps+2)
            plt.imshow(img1.squeeze().cpu().numpy(), cmap='gray')

            plt.show()








if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description="Autoencoder")
    parser.add_argument('-z', type=int, required=True)
    parser.add_argument('-e', type=int, required=True)
    parser.add_argument('-b', type=int, required=True)
    parser.add_argument('-s', type=str, required=True)
    parser.add_argument('-p', type=str, required=True)
    arguments = parser.parse_args()

    # Loading in the mnist dataset for TRAINING
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    # Defining the data loaders
    train_loader = DataLoader(train_set, arguments.b, shuffle=True)

    # Loading in the mnist dataset for TESTING
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, arguments.b, shuffle=False)

    # Set device (gpu as priority)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)

    # Import the model
    autoencoder = autoencoderMLP4Layer().to(device)
    summary(autoencoder, (1, 784))

    # Defining the optimizer
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Defining the scheduler - reduces learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    # Train the model
    train(arguments.e, optimizer, autoencoder, nn.MSELoss(), train_loader, scheduler, device, arguments.p, arguments.s)
