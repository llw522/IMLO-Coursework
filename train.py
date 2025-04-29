###Import Packages###
from multiprocessing import freeze_support
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

###Convolutional Network###
class cifar_classifier(nn.Module):      #Established outside of main() so that test.py can import class structure
    def __init__(self):
        super().__init__()              #Establish functions for forward()
        self.convert1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.convert2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.convert3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.linear1 = nn.Linear(4*4*64, 500)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(500, 10)

    def forward(self, x):               #Weighting step
        x = F.relu(self.convert1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.convert2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.convert3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*64)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

#main() established to safely import packages
def main():
    ###Normalise Data###
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #Converts data to tensor for PyTorch

    batch_size = 256    #Creates batch size for data training

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_load = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2) #Train using random subset of training data to prevent memorisation

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_load = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2) #Testing during training using same size subset as training data, also randomised

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck')

    ###Check Data Integrity###
    def imshow(image):
        image = image / 2 + 0.5
        npImage = image.numpy()
        plt.imshow(np.transpose(npImage, (1, 2, 0)))
        plt.show()

    iterData = iter(train_load)
    images, labels = next(iterData)

    imshow(torchvision.utils.make_grid(images))
    print(labels)
    print(classes)

    ###Establish Network on CPU###
    device = torch.device('cpu')
    print(device)
    classifier = cifar_classifier().to(device) #Ensure running on CPU rather than GPU

    ###Loss Function and Optimiser###
    lossFn = nn.CrossEntropyLoss()

    optimiser = optim.SGD(classifier.parameters(), lr = 0.01, momentum=0.9) #Initially coded using .Adam, but .SGD was both faster and a more successful classifier

    ###Monitored Training###
    epochs = 120 #Initial training at 240 epochs; analyis showed plataeu around the 110 mark

    losses = [] #Arrays for monitoring losses against successes
    successes = []
    test_losses = []
    test_successes = []
    total_losses = []
    total_successes = []
    test_total_losses = []
    test_total_successes = []


    for i in range(epochs):
        losses = 0.0
        successes = 0.0
        test_losses = 0.0
        test_successes = 0.0

        for inputs, labels in train_load: #Run training for epoch for all entries in batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = classifier(inputs)
            loss = lossFn(outputs, labels)

            optimiser.zero_grad() #Optimise training based on previous loss
            loss.backward()
            optimiser.step()

            _, preds = torch.max(outputs, 1) #Compare prediction to actual for current training loss
            losses += loss.item()
            successes += torch.sum(preds == labels.data)

        with torch.no_grad(): #Allows testing during training without effecting the weighting; allows monitoring to prevent memorisation
            for test_inputs, test_labels in test_load: #Test current model for all entries in batch
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                test_outputs = classifier(test_inputs)
                test_loss = lossFn(test_outputs, test_labels)

                _, test_preds = torch.max(test_outputs, 1) #Compare prediction to actual for current test loss
                test_losses += loss.item()
                test_successes += torch.sum(test_preds == test_labels.data)

        epoch_loss = losses / (len(train_load)*batch_size) #Each epoch generate average training loss
        epoch_success = successes.float() / (len(train_load)*batch_size)
        total_losses.append(epoch_loss)
        total_successes.append(epoch_success.cpu().numpy())

        test_epoch_loss = test_losses / (len(test_load)*batch_size) #Each epoch generate average test loss
        test_epoch_success = test_successes.float() / (len(test_load)*batch_size)
        test_total_losses.append(test_epoch_loss)
        test_total_successes.append(test_epoch_success.cpu().numpy())

        #Print epoch results
        print('epoch:', (i+1))
        print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_success.cpu().item()))
        print('test loss: {:.4f}, acc {:.4f} '.format(test_epoch_loss, test_epoch_success.cpu().item()))

    ###Cumulative Training Results###
    plt.style.use('ggplot')
    plt.plot(total_losses, label='Training Loss')
    plt.plot(test_total_losses, label='Test Loss')
    plt.legend() #Code programmed and functional in google collab; graph not outputted in Visual Studio

    ###Save Model###
    PATH = './cifar_net.pth'
    torch.save(classifier.state_dict(), PATH)

#Multiprocessing used to ensure code is only enacted once importing is complete
if __name__ == '__main__':
    freeze_support()
    main()