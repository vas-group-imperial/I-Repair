
"""
This file contains the logic for Influence estimation for nodes.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as tfunc

from src.models.ffnn import FFNN


class InfluenceEstimator:

    def __init__(self, model: nn.Module):

        """
        Args:
            model:
                The PyTorch Model
            input_shape:
                The input shape of the related PyTorch _model.
        """

        self._model: nn.Module = model

    # noinspection PyArgumentList
    def __call__(self, input_sets: torch.Tensor, labels: torch.Tensor, inf_type: str = "Node",
                 specific_set: bool = True) -> torch.Tensor:

        """
        Args:
            input_sets:
                A tensor of size BxNx2 with the batch in dim 1, the pixels in dim
                2 and lower and upper bound in dim 3.
            labels:
                A 1D tensor of size B with the labels corresponding to the inputs.
            inf_type:
                Either "Node" or "Weight"
            specific_set:
                If true, the influence is calculated via the cross-entropy loss,
                otherwise the absolute value.
        Returns:
                A 1D tensor of size H with the estimated influences where H is the
                number of hidden nodes in the network.
        """

        self._model.eval()

        self._calc_gradients(input_sets, labels, specific_set)

        if inf_type == "Node":
            influences = self._calc_node_influence()

        elif inf_type == "Weight":
            influences = self._calc_parameter_influence()

        else:
            raise ValueError(f"Type {inf_type} not recognised")

        return influences

    # noinspection PyArgumentList
    def _calc_node_influence(self) -> torch.Tensor:

        """
        Calculates the node influences.

        The influence is the gradients of the nodes wrt. the loss function specified
        when calling _calc_gradients.

        Returns:
            A flat tensor with the influence for each node.
        """

        influences = torch.zeros(self._model.num_nodes).to(self._model.device)

        current_idx = 0
        for layer in self._model.layers:

            if isinstance(layer, torch.nn.Linear):
                grads = torch.sum(layer.weight.grad, dim=0)
                num_nodes = grads.shape[0]
                influences[current_idx:current_idx+num_nodes] = grads
                current_idx += num_nodes

        return influences

    # noinspection PyArgumentList
    def _calc_parameter_influence(self) -> dict:

        """
        Calculates the influences of the individual parameters in the network.

        The influence is the gradients of the nodes wrt. the loss function specified
        when calling _calc_gradients.

        Returns:
            A flat tensor with the influence for each node.
        """

        influences = {}

        for i, layer in enumerate(self._model.layers):

            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) or \
                    isinstance(layer, torch.nn.BatchNorm2d):

                if layer.bias is not None:

                    influences[i] = {"weights": layer.weight.grad.cpu().detach().clone(),
                                     "biases": layer.bias.grad.cpu().detach().clone()}
                else:
                    influences[i] = {"weights": layer.weight.grad.cpu().detach().clone()}

        return influences

    # noinspection PyArgumentList,PyTypeChecker
    def _calc_gradients(self, inputs: torch.Tensor, labels: torch.Tensor, margin: bool):

        """
        Does a backward pass through the network to calculate gradients.

        Args:
            inputs:
                The inputs for which to calculate the influence.
            labels:
                The labels corresponding to the inputs.
            specific_set:
                If true, the influence is calculated via the cross-entropy loss,
                otherwise the absolute value.
        """

        for para in self._model.parameters():
            if para.grad is not None:
                para.grad.data.zero_()

        res = self._model(inputs)

        if margin:
            loss = tfunc.cross_entropy(res, labels)
        else:
            loss = torch.sum(torch.abs(res))
            loss /= inputs.shape[0]

        loss.backward()
