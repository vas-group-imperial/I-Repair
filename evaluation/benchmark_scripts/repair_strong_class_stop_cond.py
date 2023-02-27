import os
from functools import partialmethod

import torch
import torch.nn.functional as tfunc
import numpy as np

from evaluation.benchmark_scripts.util import cifar_data_loader, create_sets, write_stats
from src.repair import Repairer
from src.models.cifar_resnet import ResNet


# noinspection PyArgumentList
def _calc_margins(self, target_margin: float = 0):

    """
    Calculates margins
    """

    self._model.eval()

    pred = tfunc.softmax(self._model(self._specific_inputs), dim=1)
    argsorted_pred = torch.argsort(pred, descending=True, dim=1)

    correct_class = (argsorted_pred[:, 0] == self._specific_labels).long()
    adversarial_class = argsorted_pred[torch.arange(len(self._specific_labels)), correct_class]
    adversarial_score = pred[torch.arange(len(self._specific_labels)), adversarial_class]
    correct_score = pred[torch.arange(len(self._specific_labels)), self._specific_labels]

    margins = adversarial_score - correct_score + target_margin
    margins[margins < 0] = 0
    self._margins = margins.cpu().detach()


# noinspection PyArgumentList
def main():

    model_path = os.path.join(os.path.dirname(__file__), "../../resources/models/resnet.pth")
    resources_dir = os.path.join(os.path.dirname(__file__), "../../resources")
    results_dir = os.path.join(os.path.dirname(__file__), "../benchmark_results/repair_strong_class_stop_cond/")

    general_set_sizes = [40]
    num_repair_inputs_list = [1, 2, 4, 8, 16, 32]
    gamma = 2
    # target_margins = [0.05, 0.1, 0.15, 0.2, 0.25]
    target_margins = [0.3, 0.35, 0.4, 0.45, 0.5]

    for target_margin in target_margins:

        this_calc_margins = partialmethod(_calc_margins, target_margin=target_margin)
        Repairer._calc_margins = this_calc_margins

        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        for general_set_size in general_set_sizes:
            for num_repair_inputs in num_repair_inputs_list:

                results_file = f"{num_repair_inputs}_{target_margin}.txt"

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
                                        general_influence_mul=gamma)

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
