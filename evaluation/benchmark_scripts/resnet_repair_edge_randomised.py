import os
from functools import partialmethod

import torch
import numpy as np

from evaluation.benchmark_scripts.util import cifar_data_loader, create_sets, write_stats
from src.repair import Repairer
from src.models.cifar_resnet import ResNet
from src.repair import Repairer
from src.influence_estimation.influence_estimator import InfluenceEstimator


# noinspection PyTypeChecker
def _repair_once_edges(self, general_influence_mul: float, lock_percentage: float):
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
        rand_draw = torch.rand(specific_map[key]["weights"].shape)
        candidate_scores[key]["weights"] = torch.zeros(specific_map[key]["weights"].shape)
        candidate_scores[key]["weights"][rand_draw > lock_percentage] = 2

        if "biases" in general_map[key].keys():
            rand_draw = torch.rand(specific_map[key]["biases"].shape)
            candidate_scores[key]["biases"] = torch.zeros(specific_map[key]["biases"].shape)
            candidate_scores[key]["biases"][rand_draw > lock_percentage] = 2

    self._repair_gradient_edge(candidate_scores)


# noinspection PyArgumentList
def main():

    model_path = os.path.join(os.path.dirname(__file__), "../../resources/models/resnet.pth")
    resources_dir = os.path.join(os.path.dirname(__file__), "../../resources")
    results_dir = os.path.join(os.path.dirname(__file__), "../benchmark_results/resnet_repair_edge_randomised/")

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    general_set_sizes = [40]
    num_repair_inputs_list = [8]
    lock_percentages = ['0.9', '0.7', '0.5', '0.3', '0.1']

    for lock_percentage in lock_percentages:

        _repair_once_edges_new = partialmethod(_repair_once_edges, lock_percentage=float(lock_percentage))
        Repairer._repair_once_edges = _repair_once_edges_new

        for general_set_size in general_set_sizes:
            for num_repair_inputs in num_repair_inputs_list:

                results_file = f"{general_set_size}_{num_repair_inputs}_{lock_percentage}.txt"

                with open(os.path.join(results_dir, results_file), "w") as _:
                    pass  # Delete old file

                    for repair_class in range(10):

                        torch.manual_seed(0)

                        print(f"Running repair for class: {repair_class}")

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
                        repairer = Repairer(model, general_inputs, general_labels, specific_inputs,
                                            specific_labels, lr=1e-5)
                        repairer.repair(test_loader, max_iters=5000, influence_type="Edge",
                                        general_influence_mul=2)

                        # Save statistics to file
                        num_repaired = (repairer.margins == 0).sum()

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
