import os

import torch
import numpy as np
import torch.nn as nn

from evaluation.benchmark_scripts.util import mnist_data_loader, create_sets, write_stats
from src.repair import Repairer
from src.models.ffnn import FFNN


# noinspection PyArgumentList
def main():

    model_path = os.path.join(os.path.dirname(__file__), "../../resources/models/mnist-net_256x2.onnx")
    resources_dir = os.path.join(os.path.dirname(__file__), "../../resources")

    general_set_sizes = [40]
    num_repair_inputs_list = [1, 2, 4]
    gammas = [0.01, 0.05, 0.1, 0.5, 1]
    # gammas = [1]

    for gamma in gammas:
        results_dir = os.path.join(os.path.dirname(__file__), f"../benchmark_results/mnist_repair_edge_{gamma}/")
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        for general_set_size in general_set_sizes:
            for num_repair_inputs in num_repair_inputs_list:

                results_file = f"{general_set_size}_{num_repair_inputs}.txt"

                with open(os.path.join(results_dir, results_file), "w") as _:
                    pass  # Delete old file

                    for repair_class in range(10):

                        torch.manual_seed(0)

                        print(f"Running repair for class: {repair_class}")

                        model = FFNN.load(model_path)

                        # Import data
                        _, val_loader, test_loader, test_repair_loader = \
                            mnist_data_loader(data_dir=resources_dir, normalize=False, batch_size=128)

                        repair_data, repair_labels = next(iter(test_repair_loader))

                        # Create input sets
                        general_inputs, general_labels = create_sets(model,
                                                                     val_loader,
                                                                     general_set_size,
                                                                     classes=list(range(10)),
                                                                     correctly_classified=True,
                                                                     balanced=True)
                        specific_inputs, specific_labels = create_sets(model,
                                                                       test_loader,
                                                                       num_repair_inputs,
                                                                       classes=[repair_class],
                                                                       correctly_classified=False)

                        # Convert sets into np.array needed by InfluenceEstimator
                        specific_inputs = specific_inputs.numpy().reshape((-1, np.prod(specific_inputs.shape[1:])))
                        general_inputs = general_inputs.numpy().reshape((-1, np.prod(general_inputs.shape[1:])))
                        repair_inputs = repair_data.numpy().reshape((-1, np.prod(repair_data.shape[1:])))

                        specific_labels = specific_labels.numpy().astype(int)
                        general_labels = general_labels.numpy().astype(int)
                        repair_labels = repair_labels.numpy().astype(int)

                        # Create repairer class and call repair procedure
                        repairer = Repairer(model, general_inputs, general_labels, specific_inputs, specific_labels,
                                            lr=1e-4)
                        repairer.repair(test_loader, max_iters=5000, influence_type="Edge",
                                        general_influence_mul=gamma)

                        # Save statistics to file
                        num_repaired = (repairer.margins == 0).sum()
                        org_model = FFNN.load(model_path)

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
