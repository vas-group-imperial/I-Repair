
"""
The Neural network designed for SIP.

Author: Patrick Henriksen <patrick@henriksen.as>
"""


from typing import Callable

import torch
import torch.nn as nn
import torch.onnx as onnx

from src.util.collection import Collection


# noinspection PyTypeChecker
class FFNN(nn.Module):

    """
    The torch _model for standard FFNN networks.
    """

    def __init__(self, layers: list, out_activation: Callable = None, use_gpu: bool = False):

        """
        Args:
            layers:
                A list with the layers/ activation functions of the _model.
            out_activation:
                The activation function applied to the output.
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used
        """

        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.out_activation = out_activation
        self.device = None

        self._set_device(use_gpu=use_gpu)

        self._imap = self._build_indices_map()

    @property
    def num_nodes(self):
        num_nodes = 0
        for layer in self.layers:

            if isinstance(layer, torch.nn.Linear):
                num_nodes += layer.weight.data.shape[1]

        return num_nodes

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

    def _build_indices_map(self):

        """

        Computes a map between indices in the network and flat node list

        NOTE: I am assuming that repair may also modify outgoing connections
        of input nodes. The map will therefore include input nodes as well.

        TODO: handle convolutional layers.
        """

        start = 0
        imap = []

        for idx, l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                for node in range(l.in_features):
                    imap.append((start, (idx, node)))
                    start += 1

        return Collection(imap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward calculations for the network.

        Each object in the self.layers list is applied sequentially.

        Args:
            x:
                The input, should be BxN for FC or BxNxHxW for Conv2d, where B is the
                batch size, N is the number of nodes, H his the height and W is the
                width.

        Returns:
            The network output, of same shape as the input.
        """

        if len(x.shape) > 1:
            batch_size = x.shape[0]
        else:
            batch_size = 1

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = x.reshape(batch_size, -1)
            x = layer(x)

        x = x.reshape(batch_size, -1)

        if self.out_activation is not None:
            return self.out_activation(x)
        else:
            return x

    def save(self, path: str, dummy_input: torch.Tensor):

        """
        Saves the network to the given path in onnx format.

        Args:
            path:
                The save-path.
            dummy_input:
                A dummy input of the same dimensions as the expected input.
        """

        onnx.export(self, dummy_input, path, verbose=True, opset_version=9)

    @staticmethod
    def load(path: str):

        """
        Loads a network network in onnx format.

        Args:
             path:
                The path of the file to load.
        """

        from src.parsers.onnx_parser import ONNXParser

        onnx_parser = ONNXParser(path)
        return onnx_parser.to_pytorch()

    def model_idx_to_flat_idx(self, idx: [int]) -> int:

        tmp = tuple(idx)
        return self._imap.lookup_by_second_element(tmp)[0]

    def flat_idx_to_model_idx(self, idx: int) -> [int]:

        tmp = self._imap.lookup_by_first_element(idx)
        return list(tmp[1])
