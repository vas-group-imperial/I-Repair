import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_repair(directory: str) -> dict:

    results = {}
    files = os.listdir(directory)

    for file in files:

        file_name = file[:-4]
        results[file_name] = {}

        results[file_name]["Test"] = {}
        results[file_name]["Test"]["Acc"] = []
        results[file_name]["Test"]["Avg"] = []
        results[file_name]["Test"]["Median"] = []
        results[file_name]["Test"]["Failed"] = []

        with open(os.path.join(directory, file), "r") as f:
            for line in f:

                if line.split(" ")[0] == 'Successfully':

                    repaired = int(line.split(" ")[2].split("/")[0])
                    total = int(line.split(" ")[2].split("/")[1])
                    results[file_name]["Test"]["Failed"].append(total - repaired)

                if line[0:4] == "Test":

                    results[file[:-4]]["Test"]["Acc"].append(float(f.readline().split(" ")[-1][:-2]))

                    f.readline()
                    line = f.readline()

                    results[file[:-4]]["Test"]["Avg"].append(float(line.split(" ")[5][:-1]))
                    results[file[:-4]]["Test"]["Median"].append(float(line.split(" ")[7]))

    return results


if __name__ == '__main__':

    repair_edge = "../benchmark_results/resnet_repair_edge_2/"
    repair_edge_randomised = "../benchmark_results/resnet_repair_edge_randomised/"

    general_sizes = ['40']
    specific_sizes = ['8']
    lock_percentages = ['0.1', '0.3', '0.5', '0.7', '0.9']

    acc = []
    avg = []
    failed = []

    for results_dir in [repair_edge]:

        results = read_repair(results_dir)
        this_acc = []
        this_avg = []
        this_failed = []

        for general_size in general_sizes:
            for specific_size in specific_sizes:

                name = general_size + "_" + specific_size

                this_failed += results[name]['Test']['Failed']
                this_acc += results[name]['Test']['Acc']
                this_avg += results[name]['Test']['Avg']

        failed.append(np.array(this_failed))
        acc.append(np.array(this_acc))
        avg.append(np.array(this_avg))

    for results_dir in [repair_edge_randomised]:
        for lock_percentage in lock_percentages:

            results = read_repair(results_dir)

            this_acc = []
            this_avg = []
            this_failed = []

            for specific_size in specific_sizes:
                for general_size in general_sizes:

                    name = general_size + "_" + specific_size + '_' + lock_percentage

                    this_failed += results[name]['Test']['Failed']
                    this_acc += results[name]['Test']['Acc']
                    this_avg += results[name]['Test']['Avg']

            failed.append(np.array(this_failed))
            acc.append(np.array(this_acc))
            avg.append(np.array(this_avg))

    failed = np.array(failed)
    acc = np.array(acc) - 90.22
    avg = np.array(avg)

    acc = acc[:, failed.sum(axis=0) == 0]
    avg = avg[:, failed.sum(axis=0) == 0]

    print(failed.sum(axis=1))
    print(acc.mean(axis=1))

    avg = list(avg)
    acc = list(acc)

    fig = plt.figure()

    plt.rc('legend', fontsize=12)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    plt.boxplot(acc, showfliers=True, showmeans=True, meanline=True)
    plt.xticks([1, 2, 3, 4, 5, 6], ['I-Repair'] + lock_percentages)
    plt.xlabel("Locking probablilty")
    plt.ylabel("Accuracy change")

    lines = [Line2D([0], [0], color="orange", linestyle='-'), Line2D([0], [0], color="forestgreen", linestyle='--')]

    labels = ['Median', 'Mean']
    plt.legend(lines, labels)

    fig.tight_layout(pad=2.0)
    plt.savefig("randomised.pdf")
    plt.show()
