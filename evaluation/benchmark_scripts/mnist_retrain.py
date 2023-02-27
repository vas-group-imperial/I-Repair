import os

from src.retrain import Retrainer
from src.models.ffnn import FFNN

import torch
from evaluation.benchmark_scripts.util import mnist_data_loader, create_sets, write_stats
import numpy as np


# noinspection PyArgumentList
def main():

    model_path = os.path.join(os.path.dirname(__file__), "../../resources/models/mnist-net_256x2.onnx")
    resources_dir = os.path.join(os.path.dirname(__file__), "../../resources")
    results_dir = os.path.join(os.path.dirname(__file__), "../benchmark_results/mnist_retrain/")

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    general_set_sizes = [40]
    num_repair_inputs_list = [1, 2, 4]

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

                    # retrain model on augmented dataset
                    retrainer = Retrainer(model, general_inputs, general_labels, specific_inputs, specific_labels)
                    retrainer.retrain(test_loader, lr=1e-4, max_epochs=5000)

                    # Save statistics to file
                    num_repaired = (retrainer.margins == 0).sum()
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
