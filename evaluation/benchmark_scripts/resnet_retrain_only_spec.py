import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as tfunc
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import sampler, DataLoader
from tqdm import tqdm

from evaluation.benchmark_scripts.util import cifar_data_loader, create_sets, write_stats
from src.retrain import Retrainer
from src.models.cifar_resnet import ResNet


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

        x = self._specific_inputs.to(device=self._model.device)
        y = self._specific_labels.to(device=self._model.device)

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
def main():

    Retrainer.retrain = retrain

    model_path = os.path.join(os.path.dirname(__file__), "../../resources/models/resnet.pth")
    resources_dir = os.path.join(os.path.dirname(__file__), "../../resources")
    results_dir = os.path.join(os.path.dirname(__file__), "../benchmark_results/resnet_retrain_only_spec/")

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    general_set_sizes = [40, 80]
    num_repair_inputs_list = [1, 2, 4, 8, 16, 32]

    for general_set_size in general_set_sizes:
        for num_repair_inputs in num_repair_inputs_list:

            results_file = f"{general_set_size}_{num_repair_inputs}.txt"

            with open(os.path.join(results_dir, results_file), "w") as _:
                pass  # Delete old file

                for repair_class in range(10):

                    torch.manual_seed(0)

                    print(f"Running retraining for class: {repair_class}")

                    model = ResNet((3, 3, 3), use_gpu=True)
                    model.load(model_path)

                    # Import data
                    _, val_loader, test_loader, test_repair_loader = \
                        cifar_data_loader(data_dir=resources_dir, batch_size=128)

                    repair_data, repair_labels = next(iter(test_repair_loader))

                    # Create input sets
                    general_inputs, general_labels = create_sets(model,
                                                                 val_loader,
                                                                 general_set_size,
                                                                 classes=list(range(10)),
                                                                 correctly_classified=True,
                                                                 balanced=True)
                    specific_inputs, specific_labels = create_sets(model,
                                                                   test_repair_loader,
                                                                   num_repair_inputs,
                                                                   classes=[repair_class],
                                                                   correctly_classified=False)

                    # Convert sets into np.array needed by InfluenceEstimator
                    specific_inputs = specific_inputs.cpu().numpy()
                    general_inputs = general_inputs.cpu().numpy()
                    repair_inputs = repair_data.cpu().numpy()

                    specific_labels = specific_labels.numpy().astype(int)
                    general_labels = general_labels.numpy().astype(int)
                    repair_labels = repair_labels.numpy().astype(int)

                    # Create repairer class and call repair procedure
                    retrainer = Retrainer(model, general_inputs, general_labels, specific_inputs, specific_labels)
                    retrainer.retrain(test_loader, lr=1e-5, max_epochs=5000)

                    # Save statistics to file
                    num_repaired = (retrainer.margins == 0).sum()

                    org_model = ResNet((3, 3, 3), use_gpu=True)
                    org_model.load(model_path)

                    with open(os.path.join(results_dir, results_file), "a", buffering=1) as file:
                        write_stats(org_model,
                                    model,
                                    num_repaired,
                                    specific_inputs,
                                    specific_labels,
                                    general_inputs,
                                    general_labels,
                                    repair_inputs,
                                    repair_labels,
                                    test_loader,
                                    repair_class,
                                    file)


if __name__ == '__main__':
    main()
