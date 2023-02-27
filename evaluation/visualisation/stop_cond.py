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

    result_dirs = ["../benchmark_results/repair_strong_class_stop_cond/",
                   "../benchmark_results/retrain_strong_class_stop_cond/"]

    specific_sizes = ['4', '8', '16', '32']
    stop_conditions = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5']

    acc = []
    avg = []
    failed = []

    for results_dir in result_dirs:

        results = read_repair(results_dir)

        for specific_size in specific_sizes:

            this_acc = []
            this_avg = []
            this_failed = []

            for stop_condition in stop_conditions:

                name = specific_size + '_' + stop_condition

                failed.append(np.sum(np.array(results[name]['Test']['Failed'])))
                acc.append(np.mean(np.array(results[name]['Test']['Acc']) - 90.22))

    acc = np.array(acc)
    failed = np.array(failed)
    acc = acc.reshape((len(specific_sizes)*2, len(stop_conditions)))

    # acc = acc[:, failed.sum(axis=0) == 0]

    fig = plt.figure()

    # num_runs = len(stop_conditions)
    colors = ["dodgerblue", "orange", "limegreen", "pink"]

    plt.rc('legend', fontsize=12)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    for i in range(len(specific_sizes)):
        plt.plot(stop_conditions, acc[i], color=colors[i])
        plt.plot(stop_conditions, acc[i + len(specific_sizes)], "--", color=colors[i])

    plt.xticks(list(range(len(stop_conditions))), stop_conditions)
    plt.xlabel('Margin')
    plt.ylabel('Average accuracy drop')
    #plt.title('Strong Stopping Conditions')

    lines = [Line2D([0], [0], color="black", linestyle='-'), Line2D([0], [0], color="black", linestyle='--')]
    lines += [Line2D([0], [0], color=c, linewidth=7, linestyle='-') for c in colors]

    labels = ['Repair', 'Retrain']
    labels += [f"Repair Set Size: {num}" for num in specific_sizes]
    plt.legend(lines, labels, loc="lower left")

    #plt.legend(['Repair', 'Retrain'])

    fig.tight_layout(pad=2.0)
    plt.savefig("stop_cond.pdf")
    plt.show()
