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

#def main() established to safely import packages
def main():
    ###Normalise Data###
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_load = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_load = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

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

    ###Convolutional Network###
    class My_NN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convert1 = nn.Conv2d(3, 16, 3, 1, padding=1)
            self.convert2 = nn.Conv2d(16, 32, 3, 1, padding=1)
            self.convert3 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.linear1 = nn.Linear(4*4*64, 500)
            self.dropout = nn.Dropout(0.2)
            self.linear2 = nn.Linear(500, 10)

        def forward(self, x):
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

    ###Loss Function and Optimiser###
    ###Monitored Training###
    ###Cumulative Training Results###
    ###Save Model###

#Multiprocessing used to ensure code is only enacted once importing is complete
if __name__ == '__main__':
    freeze_support()
    main()