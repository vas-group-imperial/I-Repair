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

    dir_repair_edge_multiclass = "../benchmark_results/resnet_repair_edge_multiclass/"
    dir_retrain_multiclass = "../benchmark_results/resnet_retrain_multiclass/"

    general_sizes = ['40']
    specific_sizes = ['1', '2', '3', '4']

    acc = []
    avg = []
    failed = []

    for results_dir in [dir_repair_edge_multiclass, dir_retrain_multiclass]:

        results = read_repair(results_dir)

        for specific_size in specific_sizes:

            for general_size in general_sizes:

                this_acc = []
                this_avg = []
                this_failed = []

                name = general_size + "_" + specific_size

                this_failed += results[name]['Test']['Failed']
                this_acc += results[name]['Test']['Acc']
                this_avg += results[name]['Test']['Avg']
                failed.append(np.array(this_failed))
                acc.append(np.array(this_acc))
                avg.append(np.array(this_avg))

    failed = np.array(failed)
    print(failed)
    acc = np.array(acc) - 90.22
    print(acc)
    #acc = acc[:, failed.sum(axis=0) == 0]

    fig = plt.figure()
    plt.rc('legend', fontsize=12)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    print(acc.shape)
    plt.plot(list(range(1, 11)), acc[0:4].mean(axis=0))
    plt.plot(list(range(1, 11)), acc[4:8].mean(axis=0))

    plt.xlabel('Num classes')
    plt.ylabel('Average accuracy drop')
    plt.legend(['Repair', 'Retrain'])

    fig.tight_layout(pad=2.0)
    plt.savefig("multiclass.pdf")
    plt.show()
