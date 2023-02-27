import os

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# noinspection PyShadowingNames
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

    repair_edge = "../benchmark_results/resnet_repair_edge_stats/"
    repair_edge_randomised = "../benchmark_results/resnet_retrain_stats/"

    general_size = '40'
    specific_sizes = ['1', '2', '4', '8', '16', '32']
    lock_percentage = '0.5'
    class_nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #class_nums = ['9']

    acc = []
    avg = []
    failed = []

    for results_dir in [repair_edge]:

        results = read_repair(results_dir)

        for specific_size in specific_sizes:

            this_failed = []
            this_acc = []
            this_avg = []

            for class_num in class_nums:
                name = general_size + "_" + specific_size + "_" + class_num

                this_failed += results[name]['Test']['Failed']
                this_acc += results[name]['Test']['Acc']
                this_avg += results[name]['Test']['Avg']

            failed.append(this_failed)
            acc.append(this_acc)
            avg.append(this_avg)

    for results_dir in [repair_edge_randomised]:

        results = read_repair(results_dir)

        for specific_size in specific_sizes:

            this_failed = []
            this_acc = []
            this_avg = []

            for class_num in class_nums:
                name = general_size + "_" + specific_size + "_" + class_num

                this_failed += results[name]['Test']['Failed']
                this_acc += results[name]['Test']['Acc']
                this_avg += results[name]['Test']['Avg']

            failed.append(this_failed)
            acc.append(this_acc)
            avg.append(this_avg)

    failed = np.array(failed)
    acc = np.array(acc) - 90.22
    avg = np.array(avg)
    print(np.sum(acc > 0, axis=1))
    acc = acc[:, failed.sum(axis=0) == 0]
    print(acc.shape)
    acc = np.mean(acc.reshape((acc.shape[0], len(class_nums), 20)), axis=1)

    if failed.sum() > 0:
        print(failed.sum(axis=1))
        raise ValueError("Got failed inputs.")

    acc_avg = np.mean(acc, axis=1)

    acc = list(acc)

    conf_low = []
    conf_up = []

    for i in range(len(acc)):

        low, up = st.t.interval(0.95, len(acc[i] - 1), loc=np.mean(acc[i]), scale=st.sem(acc[i]))
        conf_low.append(low)
        conf_up.append(up)

        # bootstrap = sorted(np.mean(np.random.choice(acc[i], size=(1000, 20), replace=True), axis=1))
        # conf_low.append(bootstrap[25])
        # conf_up.append(bootstrap[975])

    for i in range(6):

        # print(acc[i])
        # print(acc[i+5])
        # print(st.wilcoxon(acc[i], acc[i+5]))
        print(st.ttest_ind(acc[i], acc[i+len(specific_sizes)], equal_var=False))

    plt.rc('legend', fontsize=12)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    fig = plt.figure()

    ticks = [1, 2, 4, 8, 16, 32]
    plt.plot(ticks, acc_avg[0:len(specific_sizes)], color="blue")
    plt.plot(ticks, conf_low[0:len(specific_sizes)], "--", color="blue")
    plt.plot(ticks, conf_up[0:len(specific_sizes)], "--", color="blue", label='_nolegend_')
    plt.fill_between(ticks, conf_low[0:len(specific_sizes)], conf_up[0:len(specific_sizes)], color="blue", alpha=0.2)

    plt.plot(ticks, acc_avg[len(specific_sizes):], color="red")
    plt.plot(ticks, conf_low[len(specific_sizes):], "--", color="red")
    plt.plot(ticks, conf_up[len(specific_sizes):], "--", color="red", label='_nolegend_')
    plt.fill_between(ticks, conf_low[len(specific_sizes):], conf_up[len(specific_sizes):], color="red", alpha=0.2)

    plt.legend(["I-Repair mean", "I-Repair 95% conf-int", "Retrain", "Retrain 95% conf-int"])

    plt.xticks(ticks)
    plt.xlabel("Repair set size")
    plt.ylabel("Accuracy change")
    fig.tight_layout(pad=2.0)
    plt.savefig("stats.pdf")

    plt.show()
