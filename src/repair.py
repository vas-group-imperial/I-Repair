
"""
The main repair procedure

Authors:
Francesco Leofante
Patrick Henriksen <patrick@henriksen.as>
"""

from typing import List

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as tfunc

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from src.influence_estimation.influence_estimator import InfluenceEstimator
from src.models.ffnn import FFNN
from src.util.logger import get_logger
from src.util.config import CONFIG

logger = get_logger(CONFIG.LOGS_LEVEL_VERIFIER, __name__, "../../logs/", "repairer_log")


class Repairer:

    # noinspection PyArgumentList
    def __init__(self,
                 model: nn.Module,
                 general_inputs: np.array,
                 general_labels: np.array,
                 specific_inputs: np.array,
                 specific_labels: np.array,
                 lr: float = 1e-4):
        """
            Args:
                model:
                    The PyTorch Model
                general_inputs:
                    Input set used to compute general influence map.
                general_labels:
                    Target classes used to compute general influence map.
                specific_inputs:
                    Input set used to compute specific influence map.
                specific_labels:
                    Target classes used to specific influence map.
                lr:
                    The learning rate used for backprop.
        """

        self._model = model

        self._specific_inputs = torch.Tensor(specific_inputs).to(self._model.device)
        self._general_labels = torch.Tensor(general_labels).long().to(self._model.device)
        self._general_inputs = torch.Tensor(general_inputs).to(self._model.device)
        self._specific_labels = torch.Tensor(specific_labels).long().to(self._model.device)

        self.lr = lr
        self._margins = None
        self._optimizer = None

    @property
    def margins(self):

        """
        Returns the margins.

        If the margins haven't been calculated yet, the calculation is performed first.
        """

        if self._margins is None:
            self._calc_margins()

        return self._margins

    # noinspection PyArgumentList
    def repair(self,
               test_dataloader: DataLoader,
               general_influence_mul: float = 10,
               influence_type: str = "Edge",
               max_iters: int = 500,
               verbose: bool = False) -> bool:

        """"
        Performs the repair procedure.

        Args:
            test_dataloader:
                Dataloader for the test set used to print statistics.
            general_influence_mul:
                The general influence is multiplied by this factor, a larger factor
                results in fewer nodes meeting the threshold for modification.
            influence_type:
                "Edge" or "Node"
            max_iters:
                The maximum number of repair iterations done.
            verbose:
                If true, statistics are printed at the end of repair.
        Returns:
            True if all repair inputs have been repaired.
        """

        self._optimizer = optim.RMSprop(self._model.parameters(), lr=self.lr)

        num_repair_inputs = len(self._specific_labels)
        remaining = self._get_remaining_misclassified()

        if len(remaining) == 0:
            return True

        pbar = tqdm(range(max_iters))
        self._update_pbar(pbar, test_dataloader, num_repair_inputs)

        i = 0
        for i in pbar:

            if influence_type == "Edge":
                self._repair_once_edges(general_influence_mul=general_influence_mul)
            elif influence_type == "Node":
                self._repair_once_node(general_influence_mul=general_influence_mul)
                pass
            else:
                raise ValueError("Influence type should be edge or node.")

            remaining = self._get_remaining_misclassified()

            self._update_pbar(pbar, test_dataloader, num_repair_inputs)

            if len(remaining) == 0:
                break

        if verbose:
            self._print_statistics(i + 1, test_dataloader)
        return len(remaining) == 0

    # noinspection PyArgumentList
    def _update_pbar(self, pbar: tqdm, dloader: DataLoader, num_repair_inputs: int):

        """
        Updates the progress bar with current statistics.

        Args:
            pbar:
                The progress bar object
            dloader:
                If provided, this is used to calculate test-set accuracy.
            num_repair_inputs:
                The original number of repair inputs.
        """

        self._model.eval()

        remaining = self._get_remaining_misclassified()

        if dloader is not None:
            correct_count = 0.0
            total_count = 0.0

            for batch_ndx, sample in enumerate(dloader):
                correct_count += (self._model(sample[0]).argmax(dim=1).cpu().detach() == sample[1]).sum()
                total_count += sample[1].size(0)
            acc = correct_count / total_count

            pbar.set_description(f"Misclassified: {len(remaining)}/{num_repair_inputs}, Test acc: "
                                 f"{acc * 100:.2f}%, Margin: {self.margins.sum():.4f}, Progress")
        else:
            pbar.set_description(f"Misclassified: {len(remaining)}/{num_repair_inputs}, "
                                 f"Margin: {self.margins.sum():.4f}, Progress")

    # noinspection PyTypeChecker
    def _repair_once_edges(self, general_influence_mul: float):

        """
        Performs one iteration of the repair procedure.

        Args:
            general_influence_mul:
                The general influence is multiplied by this factor, a larger factor
                results in fewer nodes meeting the threshold for modification.
        """

        remaining = self._get_remaining_misclassified()

        influence = InfluenceEstimator(self._model)
        general_map = influence(self._general_inputs, self._general_labels, inf_type="Weight", specific_set=False)
        specific_map = influence(self._specific_inputs[remaining], self._specific_labels[remaining],
                                 inf_type="Weight", specific_set=True)

        # Rank nodes and repair.
        candidate_scores = {}
        for key in general_map.keys():
            candidate_scores[key] = {}
            candidate_scores[key]["weights"] = ((torch.abs(specific_map[key]["weights"])+1e-8) /
                                                ((general_influence_mul * torch.abs(general_map[key]["weights"]))+1e-8))
            if "biases" in general_map[key].keys():
                candidate_scores[key]["biases"] = ((torch.abs(specific_map[key]["biases"])+1e-8) /
                                                   ((general_influence_mul * torch.abs(general_map[key]["biases"]))+1e-8))

        self._repair_gradient_edge(candidate_scores)

    def _repair_once_node(self, general_influence_mul: float = 200):

        """
        Performs one iteration of the repair procedure.

        Args:
            general_influence_mul:
                The general influence is multiplied by this factor, a larger factor
                results in fewer nodes meeting the threshold for modification.
        """

        remaining = self._get_remaining_misclassified()

        # Compute the influences for the general and specific sets
        influence = InfluenceEstimator(self._model)
        general_map = influence(self._general_inputs, self._general_labels, inf_type="Node", specific_set=False)
        specific_map = influence(self._specific_inputs[remaining], self._specific_labels[remaining],
                                 inf_type="Node", specific_set=True)

        # Rank nodes and repair.
        candidate_scores = ((torch.abs(specific_map) + 1e-8) /
                            (general_influence_mul * torch.abs(general_map) + 1e-8))

        self._repair_gradient_node(candidate_scores)

    def _repair_gradient_node(self,
                              candidate_scores: torch.Tensor,
                              l1_reg: float = 0):

        """

        Performs an iteration of the repair procedure using gradients:

        - nodes that are good candidates for repair are retrained via Adam
        - nodes that aren't are kept fixed, i.e., their gradients are masked during backprop

        Args:
            candidate_scores:
                influence scores for each node in the net
            l1_reg:
                regularisation to be used for repair

        """

        # Find indices that should be masked
        flat_idx = torch.where(candidate_scores > 1)[0].tolist()
        model_idx = [self._model.flat_idx_to_model_idx(index) for index in flat_idx]

        # Create a binary mask dict.
        masks = {}
        for layer_idx, l in enumerate(self._model.layers):
            if isinstance(l, nn.Linear):
                masks[layer_idx] = self._build_mask_node(model_idx, layer_idx, l)

        self._run_backprop(masks, l1_reg)
        self._calc_margins()

    def _build_mask_node(self, indices: List, layer_idx: int, layer: nn):

        """
        Creates gradient mask for a single layer based on influence scores

        Args:
            indices:
                list of indices identifying nodes whose grad is not to be masked
            layer_idx:
                index of layer for which mask is being created
            layer:
                torch object containing current layer
        """

        m = torch.zeros(layer.weight.shape).to(self._model.device)

        for m_idx in indices:
            if m_idx[0] == layer_idx:
                m[:, m_idx[1]] = 1

        if layer.bias is not None:
            return {"weights": m, "biases": torch.zeros(m.shape[0]).to(self._model.device)}
        else:
            return {"weights": m}

    def _repair_gradient_edge(self,
                              candidate_scores: dict,
                              l1_reg: float = 0):

        """

        Performs an iteration of the repair procedure using gradients of edges:

        - edges that are good candidates for repair are retrained via SGD
        - edges that aren't are kept fixed, i.e., their gradients are masked during backprop

        Args:
            candidate_scores:
                dictionary containing scores, one matrix for each layer with key being the index of layer
            l1_reg:
                regularisation to be used for repair

        """

        masks = {}

        for layer_idx, l in enumerate(self._model.layers):
            if isinstance(l, nn.Conv2d) or isinstance(l, torch.nn.BatchNorm2d):
                masks[layer_idx] = self._build_mask_edge(candidate_scores[layer_idx])
            elif isinstance(l, nn.Linear):
                masks[layer_idx] = self._build_mask_edge(candidate_scores[layer_idx])

        self._run_backprop(masks, l1_reg)
        self._calc_margins()

    # noinspection PyArgumentList,PyTypeChecker
    @staticmethod
    def _build_mask_edge(scores: torch.Tensor):

        """
            Creates gradient mask for a single layer based on edge influence scores

            Args:
                scores: tensor containing scores for edges in layer
        """

        if "biases" in scores.keys():
            weights = torch.where(scores["weights"] >= 1., torch.Tensor([1.]), torch.Tensor([0.]))
            biases = torch.where(scores["biases"] >= 1., torch.Tensor([1.]), torch.Tensor([0.]))
            return {"weights": weights, "biases": biases}
        else:
            weights = torch.where(scores["weights"] >= 1., torch.Tensor([1.]), torch.Tensor([0.]))
            return {"weights": weights}

    # noinspection PyArgumentList
    def _run_backprop(self,
                      masks: dict,
                      l1_reg: float):
        """
        Runs a single step of backprop for repair

        Args:
            l1_reg:
                l1 regulariser to be used
        """

        # stack all inputs
        all_data = torch.cat((self._general_inputs, self._specific_inputs), dim=0)
        all_labels = torch.cat((self._general_labels, self._specific_labels), dim=0)

        # all_data = self._specific_inputs
        # all_labels = self._specific_labels

        model = self._model.to(device=self._model.device)

        model.train()

        for layer in model.layers:
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        x = all_data.to(device=self._model.device)
        y = all_labels.to(device=self._model.device)

        scores = model(x)

        # regularization_loss = 0
        # for param in model.parameters():
        #     regularization_loss += torch.sum(torch.abs(param))

        loss = tfunc.cross_entropy(scores, y) #+ l1_reg * regularization_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._apply_masks(masks)
        self._optimizer.step()
        self._optimizer.zero_grad()
        model.eval()

    def _apply_masks(self, masks: dict):

        """
        Applies the binary-mask by multiplying with gradients of the model.

        Args:
            mask:
                A dict with binary arrays corresponding to the weights.
        """
        for layer_idx, l in enumerate(self._model.layers):
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d) or isinstance(l, torch.nn.BatchNorm2d):

                # mask gradients of weights
                masks[layer_idx]["weights"] = masks[layer_idx]["weights"].to(self._model.device)
                grads = l.weight.grad.clone()
                grads = grads * masks[layer_idx]["weights"]
                l.weight.grad = grads

                # mask gradients of bias
                if "biases" in masks[layer_idx].keys():
                    masks[layer_idx]["biases"] = masks[layer_idx]["biases"].to(self._model.device)
                    grads = l.bias.grad.clone()
                    grads = grads * masks[layer_idx]["biases"]
                    l.bias.grad = grads

    # noinspection PyArgumentList
    def _get_stats(self, data: str = "Specific"):

        """
        Returns relevant stats for the current repair datasets.

        Args:
            data:
                The dataset for which stats is calculated, should be
                "Specific" or "General".
        """

        self._model.eval()

        if data == "Specific":
            inputs = self._specific_inputs
            labels = self._specific_labels
        elif data == "General":
            inputs = self._general_inputs
            labels = self._general_labels
        else:
            raise TypeError(f"Data: {data} not implemented")

        pred = self._model(inputs)
        acc = (pred.argmax(dim=1) == labels).sum() / float(len(labels))

        return acc, pred

    # noinspection PyArgumentList
    def _calc_margins(self):

        """
        Calculates margins
        """

        self._model.eval()

        pred = self._model(self._specific_inputs)

        predicted_score = pred[torch.arange(len(self._specific_labels)), pred.argmax(dim=1)]
        correct_score = pred[torch.arange(len(self._specific_labels)), self._specific_labels]

        margins = correct_score - predicted_score

        self._margins = margins.cpu().detach()

    # noinspection PyArgumentList
    def _get_remaining_misclassified(self) -> torch.Tensor:

        """
        Finds the remaining misclassified specific inputs

        Returns:
            A tensor with the indices of misclassified points
        """

        return torch.nonzero(self.margins != 0, as_tuple=True)[0]

    # noinspection PyArgumentList
    def _print_statistics(self, iteration: int, dloader: DataLoader = None):

        """
        Prints a selection of statistics to terminal.

        Args:
            iteration:
                The current iteration
            dloader:
                If provided, this set is used to print test-set accuracy during runs.
        """

        self._model.eval()

        _, pred = self._get_stats()
        remaining = self._get_remaining_misclassified()
        num_repair_inputs = len(self._specific_labels)

        print(f"Iteration {iteration + 1}:")

        if dloader is not None:
            correct_count = 0.0
            total_count = 0.0

            for batch_ndx, sample in enumerate(dloader):
                correct_count += (self._model(sample[0]).argmax(dim=1).cpu().detach() == sample[1]).sum()
                total_count += sample[1].size(0)
            acc = correct_count / total_count

            print(f"Testset accuracy: {acc * 100:.2f}%")

        acc, _ = self._get_stats(data="General")
        print(f"General set accuracy: {acc * 100:.2f}%")
        print(f"Classifications: {pred.argmax(dim=1)}")
        print(f"Margins: {self.margins}")
        print(f"Remaining repairs: {len(remaining)}/{num_repair_inputs}")
        print("")
