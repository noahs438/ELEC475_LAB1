import datetime
import torch
import model

# COPIED FROM THE LECTURE SLIDES (as directed to in the lab)
def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    model.train()   # keep track of gradient for backtracking
    losses_train = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs in train_loader:               # imgs is a minibatch of data
            imgs = imgs.to(device=device)       # use cpu or gpu
            outputs = model(imgs)               # forward propagation through network
            loss = loss_fn(outputs, imgs)       # calculate loss
            optimizer.zero_grad()               # reset optimizer gradients to zero
            loss.backward()                     # calculate loss gradients
            optimizer.step()                    # iterate the optimization,
            loss_train += loss.item()

    scheduler.step(loss_train)                  # update some optimization hyperparameters

    losses_train += [loss_train/len(train_loader)]  # update the value of losses

    print('{} Epoch {}, Training loss {}'.format(
        datetime.datetime.now(), epoch, loss_train/len(train_loader)))

if __name__ == '__main__':
    print()