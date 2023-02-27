
"""
The main repair procedure.

Authors:
Francesco Leofante
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as tfunc
import numpy as np
from tqdm import tqdm

from src.models.ffnn import FFNN
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import sampler, DataLoader


class Retrainer:

    # noinspection PyArgumentList
    def __init__(self,
                 model: FFNN,
                 general_data: np.array,
                 general_labels: np.array,
                 specific_data: np.array,
                 specific_labels: np.array,
                 ):
        """
        Class to perform repair via retraining

        Args:
            model:
                The PyTorch Model
            general_inputs:
                The general input set.
            general_labels:
                The labels corresponding to the general inputs.
            specific_inputs:
                The specific input set.
            specific_labels:
                The labels corresponding to the specific inputs.
        """

        self._model = model

        self._general_inputs = torch.Tensor(general_data)
        self._general_labels = torch.LongTensor(general_labels)
        self._specific_inputs = torch.Tensor(specific_data)
        self._specific_labels = torch.LongTensor(specific_labels)

        self.all_data = torch.Tensor(np.vstack((general_data, specific_data)))
        self.all_labels = torch.LongTensor(np.hstack((general_labels, specific_labels)))

        self._num_repair_inputs = specific_data.shape[0]

    @property
    def margins(self):
        return self._get_margins()

    # noinspection PyArgumentList
    def retrain(self,
                test_loader: DataLoader = None,
                lr: float = 1e-3,
                l1_reg: float = 0,
                weight_decay: float = 0,
                max_epochs: int = 500,
                verbose: bool = False):

        """
        Retrains the model until all specific inputs have been classified correctly.

        Args:
            test_loader:
                The dataloader for the test set.
            lr:
                The learning rate.
            l1_reg:
                The valuemodel = FFNN.load(model_path) for the l1 regulariser.
            weight_decay:
                The value used for weight decay.
            max_epochs:
                The maximum number of epochs to train.
            verbose:
                If true, progress is printed during training.
        """

        optimizer = optim.RMSprop(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        model = self._model.to(device=self._model.device)

        pbar = tqdm(range(max_epochs))
        self._update_pbar(pbar, test_loader, self._num_repair_inputs)

        for e in pbar:

            model.train()

            if verbose:
                print(f"Dataset size: {len(self._new_train_loader)}")

            model.train()
            for layer in model.layers:
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()

            x = self.all_data.to(device=self._model.device)
            y = self.all_labels.to(device=self._model.device)

            scores = model(x)

            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))

            loss = tfunc.cross_entropy(scores, y) + l1_reg * regularization_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()

            # check if all inputs have been repaired. Stop if that's the case

            remaining = self._get_remaining_misclassified()
            self._update_pbar(pbar, test_loader, self._num_repair_inputs)

            if verbose and len(remaining) == 0:
                print(f"Repair succeeded in {e + 1} epochs")
            if len(remaining) == 0:
                break

    # noinspection PyArgumentList
    def _update_pbar(self, pbar: tqdm, test_loader: DataLoader, num_repair_inputs: int):

        """
        Updates the progress bar with current statistics.

        Args:
            pbar:
                The progress bar object
            test_loader:
                The dataloader for the test set
            num_repair_inputs:
                The original number of repair inputs.
        """

        remaining = self._get_remaining_misclassified()

        if test_loader is not None:
            correct_count = 0.0
            total_count = 0.0
            self._model.eval()

            for batch_ndx, sample in enumerate(test_loader):
                correct_count += (self._model(sample[0]).argmax(dim=1).cpu().detach() == sample[1]).sum()
                total_count += sample[1].size(0)
            acc = correct_count / total_count

            pbar.set_description(f"Misclassified: {len(remaining)}/{num_repair_inputs}, Test acc: "
                                 f"{acc * 100:.2f}%, Margin: {self.margins.sum():.4f}, Progress")
        else:
            pbar.set_description(f"Misclassified: {len(remaining)}/{num_repair_inputs}, "
                                 f"Margin: {self.margins.sum():.4f}, Progress")

    # noinspection PyArgumentList
    def _get_margins(self):

        """
        Calculates pseudo-margins using the NN predictions instead of SIP intervals.
        """

        self._model.eval()

        pred = self._model(torch.Tensor(self._specific_inputs))

        predicted_score = pred[np.arange(len(self._specific_labels)), pred.argmax(dim=1)]
        correct_score = pred[np.arange(len(self._specific_labels)), self._specific_labels.detach().numpy()]
        margins = correct_score - predicted_score

        return margins.cpu().detach().numpy()

    # noinspection PyArgumentList
    def _get_remaining_misclassified(self) -> np.array:

        """
        Finds the remaining misclassified specific inputs

        Returns:
            An array with the indices of misclassified points
        """

        return np.nonzero(self._get_margins() != 0)[0]
