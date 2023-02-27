import os

import torch
import numpy as np

from evaluation.benchmark_scripts.util import cifar_data_loader, create_sets, write_stats
from src.retrain import Retrainer
from src.models.cifar_resnet_large import ResNet


# noinspection PyArgumentList
def main():

    model_path = os.path.join(os.path.dirname(__file__), "../../resources/models/resnet_large.pth")
    resources_dir = os.path.join(os.path.dirname(__file__), "../../resources")
    results_dir = os.path.join(os.path.dirname(__file__), "../benchmark_results/resnet_large_retrain/")

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    general_set_sizes = [40, 80]
    num_repair_inputs_list = [1, 2, 4, 8, 16, 32]

    for general_set_size in general_set_sizes:
        for num_repair_inputs in num_repair_inputs_list:

            results_file = f"{general_set_size}_{num_repair_inputs}.txt"

            with open(os.path.join(results_dir, results_file), "w") as _:
                pass  # Delete old file

                for repair_class in range(10):

                    torch.manual_seed(0)

                    print(f"Running retraining for class: {repair_class}")

                    model = ResNet((5, 3, 3), use_gpu=True)
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
                    retrainer = Retrainer(model, general_inputs, general_labels, specific_inputs, specific_labels)
                    retrainer.retrain(test_loader, lr=1e-5, max_epochs=5000)

                    # Save statistics to file
                    num_repaired = (retrainer.margins == 0).sum()

                    org_model = ResNet((5, 3, 3), use_gpu=True)
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
