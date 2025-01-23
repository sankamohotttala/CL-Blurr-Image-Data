'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


from models import *
from utils import progress_bar

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

class MyCustomTransform(torch.nn.Module):# this class is only for testing
    def forward(self, img):  
        print(f"I'm transforming an image of shape {type(img)} "
            f"with bboxes = {np.array(img).shape}")
        image_array = np.array(img)
        cv2_image = image_array.copy()
        visualize_images_(img,0)
        gray_image = cv.cvtColor(cv2_image, cv.COLOR_BGR2GRAY)
        visualize_images_(gray_image,1)
        return gray_image

def visualize_images(images, num_images=10):
    plt.figure(figsize=(10, 2))  # Set the figure size
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)  # Create subplots
        plt.imshow(images[i])  # Display the image
        plt.axis('off')  # Hide axis
    plt.tight_layout()
    plt.show()

def visualize_images_(image, type_):

    plt.figure(figsize=(10, 2))  # Set the figure size
    # for i in range(num_images):
    # plt.subplot(1, num_images, i + 1)  # Create subplots
    if type_ == 0:
        plt.imshow(image)  # Display the image
    else:
        plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.tight_layout()
    plt.show()

def visualize_blur_original_imag(images_orig,images_blur, num_images=10):

    plt.figure(figsize=(10, 4))  # Set the figure size
    for i in range(num_images):
        #add subplot of 2, num_images where first row is for original image and second row is for blurred image
        plt.subplot(2, num_images, i + 1)  # Create subplots
        plt.imshow(images_orig[i])
        plt.axis('off')  # Hide axis
        plt.subplot(2, num_images, i + 1 + num_images)  # Create subplots
        plt.imshow(images_blur[i])
        plt.axis('off')  # Hide axis

    plt.tight_layout()
    plt.show()


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    MyCustomTransform(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=20, shuffle=False, num_workers=0)

for images, labels in trainloader:
    # Transformation happens here, just before the data is returned.
    print(images.shape)  # Tensor of shape (4, 3, 32, 32)
    images_np = images.numpy()  # Convert the tensor to a NumPy array
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # Reshape the array
    # visualize_images(images_np, num_images=10) 

    imag_blur = images_np.copy()

    kernal_size =3
    iterations = 3

    #blurr the images
    for _ in range(iterations):
        img_list = []
        for i in range(20):
            img = imag_blur[i]
            blur = cv.blur(img,(kernal_size,kernal_size))
            # blur = cv.GaussianBlur(img,(kernal_size,kernal_size),0)
            img_list.append(blur)
        imag_blur = np.array(img_list)

    # visualize_images(img_list, num_images=10)
    visualize_blur_original_imag(images_np,imag_blur, num_images=10)
    break

# visualize_images(trainset.data, num_images=10)  

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
