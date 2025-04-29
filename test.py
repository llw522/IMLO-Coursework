###Import Packages###
from train import cifar_classifier      #Needed to run NN
from multiprocessing import freeze_support
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#main() established to safely import packages
def main():
    ###Load Model###
    PATH = './cifar_net.pth'
    classifier = cifar_classifier()
    classifier.load_state_dict(torch.load(PATH))

    ###Normalise Data###        #Removed training data, otherwise same as in train.py
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_load = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck')

    dataiter = iter(test_load)
    images, labels = next(dataiter)

    ###Check Data Integrity###      #Same as in train.py
    def imshow(image):
        image = image / 2 + 0.5
        npImage = image.numpy()
        plt.imshow(np.transpose(npImage, (1, 2, 0)))
        plt.show()

    ###Classify Data Set###
    outputs = classifier(images)        #Run data through
    outputs.shape                       #Check data is in correct form

    ###Testing for random subset of dataset
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():       #Tests without training network
        for data in test_load:
            images, labels = data
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10): #Shows accuracy by class
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total)) #Shows accuracy for whole set

#Multiprocessing used to ensure code is only enacted once importing is complete
if __name__ == '__main__':
    freeze_support()
    main()