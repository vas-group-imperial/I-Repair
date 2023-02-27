"""
A MNIST model used for testing the verification deep_split

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as trans
import torch.nn.functional as functional

from src.models.ffnn import FFNN


# noinspection DuplicatedCode
class MNISTNN(FFNN):

    def __init__(self, layers, use_gpu: bool = False):

        """
        Args:
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used
            layers:
                The layers that should be used.
        """

        super().__init__(layers, use_gpu=use_gpu)

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None
        self.loader_train = None
        self.loader_val = None
        self.loader_test = None
        self.device = None

    def _set_device(self, use_gpu: bool):

        """
        Initializes the GPU/CPU.

        Args:
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used.
        """

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward calculations.

        Args:
            x:
                The input, should be BxN for FC or BxNxHxW for Conv2d, where B is the
                batch size, N is the number of nodes, H his the height and W is the
                width.

        Returns:
            The network output, of same shape as the input
        """

        y = super().forward(x)
        return functional.log_softmax(y, dim=1)

    def init_data_loader(self, data_dir: str, num_train: int = 49000, normalise: bool = False):

        """
        Initializes the data loaders.

        If the data isn't found, it will be downloaded.

        Args:
            data_dir:
                The directory of the data.
            num_train:
                The number of training examples used.
            normalise:
                If true, images are normalised with mean=0.1307 and std=0.3081.
        """

        if normalise:
            mean = (0.1307,)
            std = (0.3081,)
            trns_norm = trans.Compose([trans.ToTensor(), trans.Normalize(mean, std)])

        else:
            trns_norm = trans.ToTensor()

        self.mnist_train = dset.MNIST(data_dir, train=True, download=True, transform=trns_norm)
        self.loader_train = DataLoader(self.mnist_train, batch_size=64,
                                       sampler=sampler.SubsetRandomSampler(range(num_train)))

        self.mnist_val = dset.MNIST(data_dir, train=True, download=True, transform=trns_norm)
        self.loader_val = DataLoader(self.mnist_val, batch_size=64,
                                     sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

        self.mnist_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)
        self.loader_test = DataLoader(self.mnist_test, batch_size=100)

    def check_accuracy(self, loader: DataLoader) -> tuple:

        """
        Calculates and returns the accuracy of the current model.

        Args:
             loader:
                The data loader for the dataset used to calculate accuracy.
        Returns:
            (num_correct, num_samples, accuracy). The number of correct
            classifications, the total number of samples and the accuracy in percent.
        """

        num_correct = 0
        num_samples = 0

        self.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # noinspection PyCallingNonCallable
                scores = self(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            acc = float(num_correct) / num_samples

        return num_correct, num_samples, acc

    def train_model(self, epochs=10, lr=1e-3, l1_reg: float = 0, weight_decay: float = 0, verbose: bool = True):

        """
        Trains the model.

        Args:
            epochs:
                The number of epochs to train the model.
            lr:
                The learning rate
            l1_reg:
                The l1 regularization multiplier.
            weight_decay:
                The weight decay used.
            verbose:
                If true, training progress is printed.
        """

        msg = "Initialize data loaders before calling train_model"
        assert (self.loader_train is not None) and (self.loader_val is not None), msg

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        model = self.to(device=self.device)

        for e in range(epochs):

            if verbose:
                print(f"Dataset size: {len(self.loader_train)}")

            for t, (x, y) in enumerate(self.loader_train):

                model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = model(x)

                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))

                loss = functional.cross_entropy(scores, y) + l1_reg * regularization_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % 100 == 0:
                    num_correct, num_samples, acc = self.check_accuracy(self.loader_val)
                    print(f"Epoch: {e}, Iteration {t}, loss = {loss.item():.4f}")
                    print(f"Validation set results: {num_correct} / {num_samples} correct ({100 * acc:.2f})")

            if verbose:
                num_correct, num_samples, acc = self.check_accuracy(self.loader_train)
                print(f"Training set results: {num_correct} / {num_samples} correct ({100 * acc:.2f})")

        num_correct, num_samples, acc = self.check_accuracy(self.loader_test)
        print(f"Final test set results: {num_correct} / {num_samples} correct ({100 * acc:.2f})")
