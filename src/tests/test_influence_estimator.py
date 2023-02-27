"""
Unittests for the InfluenceEstimator class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import numpy as np
import unittest
import warnings

from src.tests.simple_nn import SimpleNN2
from src.influence_estimation.influence_estimator import InfluenceEstimator


# noinspection PyArgumentList,PyUnresolvedReferences,DuplicatedCode
class TestInfluenceEstimator(unittest.TestCase):

    # noinspection PyTypeChecker
    def setUp(self):

        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

        self.model = SimpleNN2(activation="Relu")
        self.influence_estimator = InfluenceEstimator(self.model, input_shape=(1,))

    def test_run_rsip(self):

        """
        Simple test of the _run_rsip method vs ground truth
        """

        concrete_bounds = np.array([[[-2., 0.]]]).astype(np.float32)
        coeffs, concrete_bounds = self.influence_estimator._run_rsip(concrete_bounds)

        gt_coeffs = np.array([[[1.125, 1, -1], [-1.125, -1, 1]]])
        gt_output_bounds = np.array([[[-2, 0], [0, 0.5], [0.5, 2.5], [-2, 0.5], [0, 2.5]]])

        for i in range(gt_coeffs.shape[1]):
            for j in range(gt_coeffs.shape[2]):
                self.assertAlmostEqual(gt_coeffs[0, i, j], coeffs[0, i, j])

        for i in range(gt_output_bounds.shape[1]):
            for j in range(gt_output_bounds.shape[2]):
                self.assertAlmostEqual(gt_output_bounds[0, i, j], concrete_bounds[0, i, j])

    def test_estimate_influences(self):

        """
        Simple test of the _estimate_influences method vs ground truth
        """

        coeffs = np.array([[[1, 2], [0, 3]]])
        labels = np.array([1])
        concrete_bounds = np.array([[[0, 2], [0, 4], [0, 2], [1, 2]]])
        influence = self.influence_estimator._estimate_influences(coeffs, labels, concrete_bounds)

        self.assertAlmostEqual(influence[0], -1)
        self.assertAlmostEqual(influence[1], 2)
