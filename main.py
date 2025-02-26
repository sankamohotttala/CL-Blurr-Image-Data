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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix

#generate random number and take the 
#take datatime and create a folder with that name

root = r'F:\Codes\joint attention\2025\pytorch-cifar\outputs'

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
folder_name  = os.path.join(root, current_time)
if not os.path.exists(folder_name): 
    os.makedirs(folder_name)

results_path = os.path.join(folder_name, 'results')
if not os.path.exists(results_path):
    # os.mkdir(results_path)
    os.makedirs(results_path)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 200

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = VGG('VGG16')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    check_path_check = os.path.join(folder_name, 'checkpoint')
    assert os.path.isdir(check_path_check), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(check_path_check,'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


train_acc_epoch = []
test_acc_epoch = []

train_acc_step = []
test_acc_step = []

train_acc_step_raw = []
test_acc_step_raw = []

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1) # returns tensors of max value and its index(class)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        step_acc_raw = predicted.eq(targets).sum().item()/targets.size(0)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
        #100.*correct/total is average acccuracy upto that batch in that epoch
        train_acc_step.append(100.*correct/total)
        train_acc_step_raw.append(step_acc_raw)

    train_acc_epoch.append(100.*correct/total) #average acc of epoch


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        predicted_list =[]
        target_list = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            predicted_list.append(predicted.cpu().numpy())
            target_list.append(targets.cpu().numpy())

            step_acc_raw = predicted.eq(targets).sum().item()/targets.size(0)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #100.*correct/total is average acccuracy upto that batch in that epoch
            test_acc_step.append(100.*correct/total)
            test_acc_step_raw.append(step_acc_raw)
        
        test_acc_epoch.append(100.*correct/total) #average acc of epoch
        #convert list of arrays to single array
        predicted_np = np.concatenate(predicted_list)
        target_np = np.concatenate(target_list)

        confusion_plot(target_np, predicted_np, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        state_new = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        check_path = os.path.join(folder_name, 'checkpoint')
        if not os.path.isdir(check_path):
            os.mkdir(check_path)
        torch.save(state, os.path.join(check_path,'ckpt.pth'))
        torch.save(state_new, os.path.join(check_path,'ckpt_opt.pth')) #new model with learning rates 
        best_acc = acc


def plot_acc(train_acc, test_acc, epoch=0, title='tmp'):
    #plot in a new figure window
    plt.figure()
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    #save the plot in the folder
    plt.savefig(os.path.join(results_path, title + '_' + str(epoch) + '.png'))
    
def confusion_plot(actual_val, predicted_val, epoch):
    
    actual_values = actual_val
    predicted_values = predicted_val

    cm = confusion_matrix(actual_values, predicted_values)
    #normalize the cm values across rows
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_epoch = np.sum(actual_values == predicted_values)/len(actual_values)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(actual_values), yticklabels=np.unique(actual_values))
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title(f'Confusion Matrix - Acc {acc_epoch:.4f}')
    # plt.show()
    #save the plot in the folder
    plt.savefig(os.path.join(results_path, 'conf' + '_' + str(epoch) + '.png'))
    

for epoch in range(start_epoch, start_epoch+num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
    plot_acc(train_acc_epoch, test_acc_epoch, epoch, 'Epoch-wise Accuracy')
    plot_acc(train_acc_step, test_acc_step, epoch,'Step-wise Accuracy')
    plot_acc(train_acc_step_raw, test_acc_step_raw, epoch,'Step-wise Accuracy (Raw)')