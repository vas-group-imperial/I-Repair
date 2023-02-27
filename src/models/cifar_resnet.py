
"""
The Neural network designed for SIP.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import random
from typing import Callable

import torch
import torch.nn as nn
import torch.onnx as onnx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import sampler
import torch.nn.functional as tfunc
import numpy as np
from tqdm import tqdm

from src.util.collection import Collection


class ResNet(nn.Module):

    def __init__(self, num_blocks: tuple, use_gpu: bool = False):

        """
        Args:
            num_blocks:
                The number of blocks (3-tuple with number of blocks of stride 1, 2, 2
                and 16, 32, 64 layers respectively).
            use_gpu:
                Determines whether to use GPU if available.
        """

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(BasicBlock, 16, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 16, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 32, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, 10)

        self.apply(self._weights_init)

        self._train_loader = None
        self._val_loader = None
        self.device = None

        self._init_dataloaders()
        self._set_device(use_gpu=use_gpu)

        self.eval()

    @property
    def layers(self):

        layers = [self.conv1, self.bn1]

        for seq in [self.layer1, self.layer2, self.layer3]:

            blocks = list(seq.children())
            for block in blocks:
                layers += list(block.children())

        layers.append(self.linear)

        return layers

    def _init_dataloaders(self,  num_train: int = 49000, data_dir: str = '../../resources/cifar/'):

        """
        Initialises the cifar data loaders.

        Args:
            num_train:
                The number of training images used.
            data_dir:
                The dataset directory.
        """

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomAffine(10),
                                          transforms.ColorJitter(brightness=0.10, contrast=0.10),
                                          transforms.RandomCrop(32, 4),
                                          transforms.ToTensor(),
                                          normalize])
        val_trans = transforms.Compose([transforms.ToTensor(), normalize])

        train_dset = datasets.CIFAR10(root=data_dir, train=True, transform=train_trans, download=False)
        val_dset = datasets.CIFAR10(root=data_dir, train=True, transform=val_trans, download=False)

        self._train_loader = torch.utils.data.DataLoader(train_dset, batch_size=128,
                                                         sampler=sampler.SubsetRandomSampler(range(num_train)))
        self._val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1000,
                                                       sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

    def _set_device(self, use_gpu: bool):

        """
        Initializes the gpu/ cpu

        Args:
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used
        """

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.to(device=self.device)

    def forward(self, x: torch.Tensor):

        """
        Calculates the network output.

        Args:
            x:
                The input.
        """

        x = x.to(self.device)

        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)

        out = tfunc.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = tfunc.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    @staticmethod
    def _make_layer(block: Callable, in_channels: int, channels: int, num_blocks: int, stride: int):

        """
        Makes a ResNet layer
        
        Args:
            block:
                The block class
            in_channels:
                Number of in channels
            channels:
                Number of channels in the block
            num_blocks:
                The number of blocks in the layer
            stride:
                The stride used for the first block in the layer.

        Returns:
            A sequential object with the blocks.
        """

        layers = [block(in_channels, channels, stride)]

        for i in range(num_blocks - 1):
            layers.append(block(channels, channels, 1))

        return nn.Sequential(*layers)

    @staticmethod
    def _weights_init(m):

        """
            Initialization of CNN weights
        """

        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)

    def train_model(self, optimiser: torch.optim, lr_scheduler: torch.optim.lr_scheduler = None, epochs: int = 100):

        """
        Trains the model

        Args:
            optimiser:
                The optimiser to be used.
            lr_scheduler:
                The scheduler.
            epochs:
                The number of epochs to train for.
        """

        pbar = tqdm(range(epochs))
        pbar.set_description(f"Train loss: -, Train acc: -, "
                             f"Val acc: {self.validation_accuracy():.2f}, Progress")

        for _ in pbar:

            train_loss, train_acc = self._train_epoch(optimiser, lr_scheduler)
            pbar.set_description(f"Train loss: {train_loss:.2f}, Train acc: {train_acc:.2f}, "
                                 f"Val acc: {self.validation_accuracy():.2f}, Progress")

    def _train_epoch(self, optimiser: torch.optim, lr_scheduler: torch.optim.lr_scheduler) -> tuple:

        """
        Run one train epoch

        Args:
            optimiser:
                The optimiser to be used.
            lr_scheduler:
                The scheduler.
        """

        self.train()

        loss = None
        correct = 0
        total = 0

        for i, (x, y) in enumerate(self._train_loader):

            y = y.to(self.device)
            x = x.to(self.device)

            output = self(x)
            correct += torch.sum((torch.argmax(output, dim=1) == y))
            total += y.shape[0]

            loss = torch.nn.functional.cross_entropy(output, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            loss = loss.float()

        acc = 100*correct/total

        self.eval()

        return float(loss), float(acc)

    def validation_accuracy(self, subset: bool = True) -> float:

        """
        Calculates the validation accuracy for one batch from the validation set.

        subset:
            If true, a random subset of the validation set is used
        """

        self.eval()

        if subset:
            x, y = next(iter(self._val_loader))
            x = x.to(self.device)
            y = y.to(self.device)
            output = self(x)

            acc = float(100 * torch.sum((torch.argmax(output, dim=1) == y)) / y.shape[0])

        else:

            correct = 0
            total = 0

            for x, y in self._val_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                output = self(x)

                correct += torch.sum((torch.argmax(output, dim=1) == y))
                total += y.shape[0]

            acc = float(100 * correct / total)

        return acc

    def save(self, path: str = "../../resources/models/resnet.pth"):

        """
        Saves the network to the given path in onnx format.

        Args:
            path:
                The save-path.
        """

        torch.save(self.state_dict(), path)

    def load(self, path: str = "../../resources/models/resnet.pth"):

        """
        Loads a network network in onnx format.

        Args:
             path:
                The path of the file to load.
        """

        self.load_state_dict(torch.load(path, map_location=self.device))


class LambdaLayer(nn.Module):

    def __init__(self, lambd):

        """
        Args:
            lambd:
                The function to be applied
        """

        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):

        """
        Applies the self.lambd function.
        """

        return self.lambd(x)


class BasicBlock(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, in_channels, channels, stride=1):

        """
        Args:
            in_channels:
                The number of input channels to the block.
            channels:
                The number of channels used in the block.
            stride:
                The stride of the first convolution.
        """

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != channels:
            self.shortcut = LambdaLayer(lambda x: tfunc.pad(x[:, :, ::2, ::2],
                                                            (0, 0, 0, 0, channels // 4, channels // 4), "constant", 0))

    def forward(self, x):

        """
        Applies the block layers to the input.

        Args:
            x:
                The input
        """

        out = tfunc.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tfunc.relu(out)

        return out
