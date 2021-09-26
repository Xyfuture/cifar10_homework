from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from collections import OrderedDict


def download_cifar10(save_path):
    torchvision.datasets.CIFAR10(root=save_path,train=True,download=True)
    torchvision.datasets.CIFAR10(root=save_path,train=False,download=True)
    return save_path

def load_cifar10(batch_size=64,pth_path='./',img_size=32):
    train_trans_list = [transforms.Resize((img_size,img_size)),
                        transforms.Pad(padding=4),
                        transforms.RandomCrop(img_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]
    test_trans_list = [transforms.Resize((img_size,img_size)),
                        transforms.ToTensor()]
    train_transform = transforms.Compose(train_trans_list)
    test_transform = transforms.Compose(test_trans_list)
    # if img_size!=32:
    #     transform = transforms.Compose(
    #         [transforms.Resize((img_size,img_size)),
    #         transforms.ToTensor()])
    #     test_transform = transforms.Compose([transforms.Resize((img_size,img_size))
    #         ,transforms.ToTensor()])
    # else:
    #     transform = transforms.Compose([transforms.Pad(padding = 4),
    #         transforms.RandomCrop(32),
    #         transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    #     test_transform = transforms.Compose([transforms.ToTensor()])
   
    trainset = torchvision.datasets.CIFAR10(root=pth_path, train=True,download=False, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=pth_path, train=False,download=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    dataloaders = {"train":trainloader,"val":testloader}
    dataset_sizes = {"train":50000,"val":10000}
    return dataloaders,dataset_sizes



def train_model(model, dataloaders, dataset_sizes , criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.cuda()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

def test_model(model,dataloaders,dataset_sizes,criterion):
    print("validation model:")
    phase = "val"
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0
        for inputs,labels in tqdm(dataloaders[phase]):
            if torch.cuda.is_available():
                inputs,labels = inputs.cuda(),labels.cuda()
            outputs = model(inputs)
            _,preds = torch.max(outputs,1)
            loss = criterion(outputs,labels)
            running_loss += loss.item() * inputs.size(0)
            running_acc += torch.sum(preds == labels.data)
        epoch_loss = running_loss/dataset_sizes[phase]
        epoch_acc = running_acc / dataset_sizes[phase]
        epoch_acc = epoch_acc.item()
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
    return epoch_acc,epoch_loss

def eval_model(model,dataloaders,dataset_sizes,criterion):
    print('evaluating the model')
    if torch.cuda.is_availabel():
        model.cuda()
    model.eval()
    for phase in ['train','val']:
        running_loss = 0.0
        running_acc = 0.0
        with torch.no_grad():
            for inputs,labels in tqdm(dataloaders[phase]):
                if torch.cuda.is_available():
                    inputs,labels = inputs.cuda(),labels.cuda()
                outputs = model(inputs)
                _,preds = torch.max(inputs)
                loss = criterion(outputs,labels)
                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)
            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_acc = epoch_acc.item()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
    

        
