
"""
Unittests for the layers mappings

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest

import numpy as np
import torch
import torch.nn.functional as tf

from src.influence_estimation.mappings.layers import FC, Conv2d, BatchNorm2d


class TestMappingLayers(unittest.TestCase):

    def setUp(self):

        self.fc = FC()
        self.fc.params["weight"] = np.array([[1, 2, 3], [4, 5, 6]])

        self.fc.params["bias"] = np.array([1, 2])

        self.conv_2d = Conv2d()
        self.conv_2d.params["weight"] = np.array([[[[1, 1, 1], [1, 1, 0], [1, 0, 0]]]])
        self.conv_2d.params["bias"] = np.array([1])
        self.conv_2d.params["stride"] = (1, 1)
        self.conv_2d.params["padding"] = (1, 1)
        self.conv_2d.params["out_channels"] = 1
        self.conv_2d.params["kernel_size"] = (3, 3)
        self.conv_2d.params["in_shape"] = (1, 2, 2)

        self.batch_norm_2d = BatchNorm2d()
        self.batch_norm_2d.params["weight"] = np.array([2])
        self.batch_norm_2d.params["bias"] = np.array([1])
        self.batch_norm_2d.params["running_mean"] = np.array([1])
        self.batch_norm_2d.params["running_var"] = np.array([2])
        self.batch_norm_2d.params["in_shape"] = (1, 2, 2)
        self.batch_norm_2d.params["eps"] = 0.1

    def test_fc_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertFalse(self.fc.is_1d_to_1d)
        self.assertTrue(self.fc.is_linear)

    def test_fc_propagate(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([[-1, -2, -3]]).T
        res = self.fc.propagate(x, add_bias=True)

        self.assertAlmostEqual(res[0, 0], -13)
        self.assertAlmostEqual(res[1, 0], -30)

    def test_fc_propagate_transposed(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([[0.5, 1, 0.5], [0, 1, 0]], dtype=np.float32)
        self.fc.params["weight"] = np.array([[1, -1], [1, 1]], dtype=np.float32)
        self.fc.params["bias"] = np.array([1, 1], dtype=np.float32)

        res = self.fc.propagate_reversed(x, add_bias=True)

        self.assertAlmostEqual(res[0, 0], 1.5)
        self.assertAlmostEqual(res[0, 1], 0.5)
        self.assertAlmostEqual(res[0, 2], 2)
        self.assertAlmostEqual(res[1, 0], 1)
        self.assertAlmostEqual(res[1, 1], 1)
        self.assertAlmostEqual(res[1, 2], 1)

    def test_conv_2d_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertFalse(self.conv_2d.is_1d_to_1d)
        self.assertTrue(self.conv_2d.is_linear)

    def test_conv_2d_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([[-1, -2, -3, -4]]).T
        gt = np.array([[0, -5, -5, -9]]).T
        res = self.conv_2d.propagate(x, add_bias=True)

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)

        gt = np.array([[-1, -6, -6, -10]]).T
        res = self.conv_2d.propagate(x, add_bias=False)

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)

    def test_batch_norm_2d_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertFalse(self.batch_norm_2d.is_1d_to_1d)
        self.assertTrue(self.batch_norm_2d.is_linear)

    def test_batch_norm_2d_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([[-1, -2, -3, -4]]).T

        res = self.batch_norm_2d.propagate(x, add_bias=True)
        gt = tf.batch_norm(input=torch.Tensor(x),
                           running_mean=torch.Tensor(self.batch_norm_2d.params["running_mean"]),
                           running_var=torch.Tensor(self.batch_norm_2d.params["running_var"]),
                           weight=torch.Tensor(self.batch_norm_2d.params["weight"]),
                           bias=torch.Tensor(self.batch_norm_2d.params["bias"]),
                           eps=self.batch_norm_2d.params["eps"])

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)

    def test_reversed_conv2(self):

        """
        Test the reverse propagation used by RSIP vs manually calculated gt.
        """

        self.conv_2d.params["weight"] = np.array([[[[2, 1, 1], [1, 2, 1], [1, 1, 2]],
                                                   [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
                                                   [[2, 1, 1], [1, 2, 1], [1, 1, 2]]],
                                                 [[[4, 2, 2], [2, 4, 2], [2, 2, 4]],
                                                  [[4, 2, 2], [2, 4, 2], [2, 2, 4]],
                                                  [[4, 2, 2], [2, 4, 2], [2, 2, 4]]]])

        self.conv_2d.params["bias"] = np.array([1., 2.], dtype=np.float32)
        self.conv_2d.params["stride"] = (1, 1)
        self.conv_2d.params["padding"] = (1, 1)
        self.conv_2d.params["out_channels"] = 2
        self.conv_2d.params["kernel_size"] = (3, 3)
        self.conv_2d.params["in_shape"] = (3, 2, 2)

        x = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 0, 0, 0, 0, 1],
                      [-1, 1, 0, 0, 0, 0, 0, 0, 2],
                      [1, 0, -1, 1, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
        x_new = self.conv_2d.propagate_reversed(x)
        x_gt = np.array([[2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1],
                         [1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
                         [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1],
                         [2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1],
                         [6, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 6, 5],
                         [-1, 1, 0, -1, -1, 1, 0, -1, -1, 1, 0, -1, 2],
                         [3, 1, 0, 3, 3, 1, 0, 3, 3, 1, 0, 3, 2],
                         [18, 15, 15, 18, 18, 15, 15, 18, 18, 15, 15, 18, 13]], dtype=np.float32)

        for i in range(len(x_gt)):
            for j in range(len(x_gt[i])):
                self.assertAlmostEqual(x_new[i, j], x_gt[i, j])

    def test_reversed_batch_norm_2d(self):

        """
        Test the reverse propagation used by RSIP vs manually calculated gt.
        """

        self.batch_norm_2d.params["weight"] = np.array([1, 4])
        self.batch_norm_2d.params["bias"] = np.array([3, 4])
        self.batch_norm_2d.params["running_mean"] = np.array([2, 2])
        self.batch_norm_2d.params["running_var"] = np.array([1.1, 4.1])
        self.batch_norm_2d.params["post_shape"] = (2, 2, 2)
        self.batch_norm_2d.params["eps"] = -0.1

        x = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [2, 2, 2, 2, 2, 2, 2, 2, 2],
                      [0, 0, 0, 0, 0, 0, 3, 0, 3]], dtype=np.float32)

        x_new = self.batch_norm_2d.propagate_reversed(x)

        x_gt = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 5],
                         [0, 1, 0, 0, 0, 0, 0, 0, 5],
                         [0, 0, 1, 0, 0, 0, 0, 0, 5],
                         [0, 0, 0, 1, 0, 0, 0, 0, 5],
                         [1, 0, 0, 0, 2, 2, 2, 2, 38],
                         [0, 1, 1, 1, 2, 2, 2, 0, 39],
                         [2, 2, 2, 2, 4, 4, 4, 4, 106],
                         [0, 0, 0, 0, 0, 0, 6, 0, 27]], dtype=np.float32)

        for i in range(len(x_gt)):
            for j in range(len(x_new)):
                self.assertAlmostEqual(x_new[i, j], x_gt[i, j])
