import os

import numpy as np
import matplotlib.pyplot as plt


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

    result_dirs = ["../benchmark_results/resnet_retrain/",
                   "../benchmark_results/resnet_repair_edge_0.001/",
                   "../benchmark_results/resnet_repair_edge_0.01/",
                   "../benchmark_results/resnet_repair_edge_0.05/",
                   "../benchmark_results/resnet_repair_edge_0.1/",
                   "../benchmark_results/resnet_repair_edge_0.5/",
                   "../benchmark_results/resnet_repair_edge_1/",
                   "../benchmark_results/resnet_repair_edge_2/"
    ]

    general_sizes = ['40', '80']
    specific_sizes = ['1', '2', '4', '8', '16', '32']

    acc = []
    avg = []
    failed = []

    for results_dir in result_dirs:

        results = read_repair(results_dir)

        for specific_size in specific_sizes:

            this_acc = []
            this_avg = []
            this_failed = []

            for general_size in general_sizes:

                name = general_size + "_" + specific_size

                this_failed += results[name]['Test']['Failed']
                this_acc += results[name]['Test']['Acc']
                this_avg += results[name]['Test']['Avg']

            failed.append(np.array(this_failed))
            acc.append(np.array(this_acc))
            avg.append(np.array(this_avg))

    failed = np.array(failed)
    acc = np.array(acc) - 90.22

    acc = acc[:, failed.sum(axis=0) == 0]

    print(failed.sum(axis=1))
    print(acc.mean(axis=1))

    fig = plt.figure()

    num_runs = len(specific_sizes)
    for i in range(len(result_dirs)):
        plt.plot(list(range(num_runs)), acc.mean(axis=1)[i*num_runs:(i+1)*num_runs])

    plt.xticks(list(range(len(specific_sizes))), specific_sizes)
    plt.xlabel('Repair set size')
    plt.ylabel('Average accuracy drop')
    plt.title('ResNet (200k ReLUs) Repair vs Retrain')
    plt.legend(['Retrain (Gamma=0)', 'Gamma=0.001', 'Gamma=0.01', 'Gamma=0.05', 'Gamma=0.1', 'Gamma=0.5', 'Gamma=1', 'Gamma=2', 'Gamma=5'])

    fig.tight_layout(pad=2.0)
    # plt.savefig("resnet.pdf")
    plt.show()
