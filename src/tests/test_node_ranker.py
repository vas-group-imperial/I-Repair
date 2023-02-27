"""
Unittests for the NodeRanker class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numpy as np
import unittest
import warnings

from src.node_ranking import NodeRanker


# noinspection PyArgumentList,PyUnresolvedReferences,DuplicatedCode
class TestNodeRanker(unittest.TestCase):

    # noinspection PyTypeChecker
    def setUp(self):

        self.ranker = NodeRanker()

    def test_get_pareto_front(self):

        """
        Simple test of the _pareto_front method vs ground truth
        """

        ranker = NodeRanker()
        values = np.abs(np.vstack((np.array([1, 2, 3, 0, 8, -1]), np.array([2, 3, 4, 5, 1, -1]))).T)
        res = ranker._get_pareto_front(np.vstack(values))
        gt = [2, 3, 4]

        for i in range(len(gt)):
            self.assertEqual(res[i], gt[i])

    def test_get_compromise(self):

        """
        Simple test of the _get_pareto_compromise method vs ground truth
        """

        ranker = NodeRanker()
        values = np.abs(np.vstack((np.array([2, 0.1, 1]), np.array([0, 2, 1]))).T)
        res = ranker._get_compromise(values)

        gt = [2, 1, 0]
        for i in range(len(gt)):
            self.assertEqual(res[i], gt[i])

    def test__call__(self):

        """
        Simple test of the __call__ method vs ground truth
        """

        ranker = NodeRanker()
        general_influence, specific_influence = np.array([-0.5, -2, 0.1, 0.8, 0.8]), np.array([0.2, 0, 2, 1, 0.8])
        res = ranker(general_influence, specific_influence)
        gt = [3, 4, 2]
        for i in range(len(gt)):
            self.assertEqual(res[i], gt[i])
