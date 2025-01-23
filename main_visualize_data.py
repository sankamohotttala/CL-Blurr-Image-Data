'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def visualize_images(images, num_images=10):
    """
    Visualize a specified number of images from a 4D NumPy array.
    
    Args:
    images (numpy.ndarray): Image data of shape (num_samples, height, width, channels).
    num_images (int): Number of images to display.
    """
    plt.figure(figsize=(10, 2))  # Set the figure size
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)  # Create subplots
        plt.imshow(images[i])  # Display the image
        plt.axis('off')  # Hide axis
    plt.tight_layout()
    plt.show()



# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=0)

for images, labels in trainloader:
    # Transformation happens here, just before the data is returned.
    print(images.shape)  # Tensor of shape (4, 3, 32, 32)
    images_np = images.numpy()  # Convert the tensor to a NumPy array
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # Reshape the array
    visualize_images(images_np, num_images=10)  
    break

# visualize_images(trainset.data, num_images=10)  

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
