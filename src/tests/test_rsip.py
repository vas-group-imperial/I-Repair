
"""
Unittests for the RSIP class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import numpy as np
import unittest
import warnings

from src.tests.simple_nn import SimpleNN, SimpleNNConv2, SimpleNNBatchNorm2D
from src.influence_estimation.rsip import RSIP
from src.influence_estimation.mappings.piecewise_linear import Relu
from src.influence_estimation.mappings.s_shaped import Sigmoid, Tanh
from src.influence_estimation.mappings.layers import FC, Conv2d, BatchNorm2d


# noinspection PyArgumentList,PyUnresolvedReferences,DuplicatedCode
class TestRSIP(unittest.TestCase):

    # noinspection PyTypeChecker
    def setUp(self):

        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

        self.model_sigmoid = SimpleNN(activation="Sigmoid")
        self.model_tanh = SimpleNN(activation="Tanh")
        self.model_relu = SimpleNN(activation="Relu")
        self.model_conv2 = SimpleNNConv2()
        self.model_batch_norm2d = SimpleNNBatchNorm2D()

        self.bounds_sigmoid = RSIP(self.model_sigmoid, input_shape=2, optimise_computations=False)
        self.bounds_tanh = RSIP(self.model_tanh, input_shape=2, optimise_computations=False)
        self.bounds_relu = RSIP(self.model_relu, input_shape=2, optimise_computations=False)
        self.bounds_conv2 = RSIP(self.model_conv2, input_shape=(1, 5, 5), optimise_computations=False)
        self.bounds_batch_norm_2d = RSIP(self.model_batch_norm2d, input_shape=(1, 2, 2), optimise_computations=False)

    # noinspection PyTypeHints
    def test_mappings_initialization(self):

        """
        Test that assigned mappings are correct.
        """

        mappings_relu = [None, FC, Relu, FC]
        for i in range(1, len(mappings_relu)):
            self.assertTrue(isinstance(self.bounds_relu._mappings[i], mappings_relu[i]))

        mappings_sigmoid = [None, FC, Sigmoid, FC]
        for i in range(1, len(mappings_sigmoid)):
            self.assertTrue(isinstance(self.bounds_sigmoid._mappings[i], mappings_sigmoid[i]))

        mappings_tanh = [None, FC, Tanh, FC]
        for i in range(1, len(mappings_tanh)):
            self.assertTrue(isinstance(self.bounds_tanh._mappings[i], mappings_tanh[i]))

        mappings_conv2 = [None, Conv2d]
        for i in range(1, len(mappings_conv2)):
            self.assertTrue(isinstance(self.bounds_conv2._mappings[i], mappings_conv2[i]))

        mappings_batch_norm_2d = [None, Conv2d, BatchNorm2d]
        for i in range(1, len(mappings_batch_norm_2d)):
            self.assertTrue(isinstance(self.bounds_batch_norm_2d._mappings[i], mappings_batch_norm_2d[i]))

    def test_adjust_bounds_from_forced_bounds(self):

        """
        Tests the output from the _adjust_bounds_from_forced_bounds() method against
        ground truth.
        """

        concrete_bounds = np.array([[1., 2.], [-2., 0.]]).astype(np.float32)
        forced_bounds = np.array([[-10, 1], [-1, 10]]).astype(np.float32)
        gt_bounds = np.array([[1., 1.], [-1., 0.]]).astype(np.float32)
        concrete_bounds = self.bounds_relu._adjust_bounds_from_forced(concrete_bounds, forced_bounds)

        for node_num in range(2):
            self.assertAlmostEqual(gt_bounds[node_num, 0],
                                   concrete_bounds[node_num, 0])
            self.assertAlmostEqual(gt_bounds[node_num, 1],
                                   concrete_bounds[node_num, 1])

    def test_valid_concrete_bounds(self):

        """
        Tests the output from the _valid_concrete_bounds() method against ground truth.
        """

        concrete_bounds_valid = np.array([[1, 2], [-1, 2], [3, 5]]).astype(np.float32)
        concrete_bounds_invalid = np.array([[1, -1], [-1, 2], [3, 5]]).astype(np.float32)

        self.assertTrue(self.bounds_relu._valid_concrete_bounds(concrete_bounds_valid))
        self.assertFalse(self.bounds_relu._valid_concrete_bounds(concrete_bounds_invalid))

    def test_calc_relaxations(self):

        """
        Tests the output from the _calc_relaxation() method against ground truth.
        """

        concrete_bounds_low = np.array([[-2, -1], [-1, 1], [3, 5]]).astype(np.float32)
        concrete_bounds_up = np.array([[-1, -0.5], [-1, 1], [6, 7]])

        gt_relax_lower = np.array([[0, 0], [0, 0], [1, 0]]).astype(np.float32)
        gt_relax_upper = np.array([[0, 0], [0.5, 0.5], [1, 0]]).astype(np.float32)

        calc_relax = self.bounds_relu._calc_relaxations(Relu(), concrete_bounds_low, concrete_bounds_up)

        for i in range(gt_relax_upper.shape[0]):
            for j in range(gt_relax_upper.shape[1]):
                self.assertAlmostEqual(gt_relax_lower[i, j], calc_relax[0, i, j])
                self.assertAlmostEqual(gt_relax_upper[i, j], calc_relax[1, i, j])

    def test_prop_equation_trough_relaxation_lower(self):

        """
        Tests the output from the _prop_equation_trough_relaxation() method against
        ground truth.
        """

        symb_bounds = np.array([[1, 2, 3], [-1, 0.5, 2]]).astype(np.float32)
        relax = np.array(([[[0, 0], [0.3, 0.4]], [[0.5, 0.5], [1, 0]]])).astype(np.float32)

        gt_new_symb_bounds = np.array([[0, 0.6, 3.8], [-0.5, 0.15, 1.7]]).astype(np.float32)
        res = self.bounds_relu._backprop_through_relaxation(symb_bounds, relax, lower=True)

        for i in range(gt_new_symb_bounds.shape[0]):
            for j in range(gt_new_symb_bounds.shape[1]):
                self.assertAlmostEqual(res[i, j], gt_new_symb_bounds[i, j])

    def test_prop_equation_trough_relaxation_upper(self):

        """
        Tests the output from the _prop_equation_trough_relaxation() method against
        ground truth.
        """

        symb_bounds = np.array([[1, 2, 3], [-1, 0.5, 2]]).astype(np.float32)
        relax = np.array(([[[0, 0], [0.3, 0.4]], [[0.5, 0.5], [1, 0]]])).astype(np.float32)

        gt_new_symb_bounds = np.array([[0.5, 2, 3.5], [0, 0.5, 2]]).astype(np.float32)
        res = self.bounds_relu._backprop_through_relaxation(symb_bounds, relax, lower=False)

        for i in range(gt_new_symb_bounds.shape[0]):
            for j in range(gt_new_symb_bounds.shape[1]):
                self.assertAlmostEqual(res[i, j], gt_new_symb_bounds[i, j])

    def test_calculate_symbolic_bounds_sigmoid_brute_force(self):

        """
        Tests that the neural network is within calculated bounds for a range of symb_bounds_in
        values.
        """

        x1_range = [-0.5, 1]
        x2_range = [-0.2, 0.6]
        input_constraints = np.array([[x1_range[0], x1_range[1]], [x2_range[0], x2_range[1]]]).astype(np.float32)

        x1_arr = np.linspace(x1_range[0], x1_range[1], 100).astype(np.float32)
        x2_arr = np.linspace(x2_range[0], x2_range[1], 100).astype(np.float32)

        self.bounds_sigmoid.calc_bounds(input_constraints)
        bound_symb = self.bounds_sigmoid.bounds_concrete

        # Do a brute force check with different symb_bounds_in values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.model_sigmoid.forward(torch.Tensor([[x1, x2]])).detach().numpy()[0, 0]
                self.assertLessEqual(bound_symb[-1][:, 0], res)
                self.assertGreaterEqual(bound_symb[-1][:, 1], res)

    def test_calculate_symbolic_bounds_tanh_brute_force(self):

        """
        Tests that the neural network is within calculated bounds for a range of symb_bounds_in
        values.
        """

        x1_range = [-0.5, 1]
        x2_range = [-0.2, 0.6]
        input_constraints = np.array([[x1_range[0], x1_range[1]], [x2_range[0], x2_range[1]]]).astype(np.float32)

        x1_arr = np.linspace(x1_range[0], x1_range[1], 100).astype(np.float32)
        x2_arr = np.linspace(x2_range[0], x2_range[1], 100).astype(np.float32)

        self.bounds_tanh.calc_bounds(input_constraints)
        bound_symb = self.bounds_tanh.bounds_concrete

        # Do a brute force check with different symb_bounds_in values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.model_tanh.forward(torch.Tensor([[x1, x2]])).detach().numpy()[0, 0]
                self.assertLessEqual(bound_symb[-1][:, 0], res)
                self.assertGreaterEqual(bound_symb[-1][:, 1], res)

    def test_calculate_symbolic_bounds_relu_brute_force(self):

        """
        Tests that the neural network is within calculated bounds for a range of symb_bounds_in
        values.
        """

        x1_range = [-1, 1]
        x2_range = [-2, 2]
        input_constraints = np.array([[x1_range[0], x1_range[1]], [x2_range[0], x2_range[1]]]).astype(np.float32)

        x1_arr = np.linspace(x1_range[0], x1_range[1], 100).astype(np.float32)
        x2_arr = np.linspace(x2_range[0], x2_range[1], 100).astype(np.float32)

        self.bounds_relu.calc_bounds(input_constraints)
        bound_symb = self.bounds_relu.bounds_concrete

        # Do a brute force check with different symb_bounds_in values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.model_relu.forward(torch.Tensor([[x1, x2]])).detach().numpy()[0, 0]
                self.assertLessEqual(bound_symb[-1][:, 0], res)
                self.assertGreaterEqual(bound_symb[-1][:, 1], res)
