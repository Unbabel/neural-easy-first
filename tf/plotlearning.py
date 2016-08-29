import argparse
import codecs
import re
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    plot_dict = dict()
    for logfile in args.log_files:
        with codecs.open(logfile, "r", "utf8") as l:
            train_F1_OKs, dev_F1_OKs = [], []
            train_F1_BADs, dev_F1_BADs = [], []
            train_F1_prods, dev_F1_prods = [], []
            epochs = 0
            for line in l:
                if "EPOCH" in line:
                    if "train" in line:
                        epochs += 1
                        F1_OK, F1_BAD = re.findall("\((.*)/(.*)\)", line)[0]
                        train_F1_BADs.append(float(F1_BAD))
                        train_F1_OKs.append(float(F1_OK))
                        train_F1_prods.append(float(F1_OK)*float(F1_BAD))
                    elif "dev" in line:
                        F1_OK, F1_BAD = re.findall("\((.*)/(.*)\)", line)[0]
                        dev_F1_BADs.append(float(F1_BAD))
                        dev_F1_OKs.append(float(F1_OK))
                        dev_F1_prods.append(float(F1_OK)*float(F1_BAD))
            plot_dict[logfile.split(".log")[0]] = (train_F1_OKs, train_F1_BADs, train_F1_prods)

    subplots = len(plot_dict)
    min_epoch = min([len(data[0]) for data in plot_dict.values()])
    ymin = min([min(data[2]) for data in plot_dict.values()])
    ymax = max([max(data[0]) for data in plot_dict.values()])
    i = 0
    for run, data in plot_dict.items():
        i += 1
        plt.subplot(subplots, 1, i)
        x = np.arange(len(data[0]))
        y_OK = data[0]
        y_BAD = data[1]
        y_prod = data[2]
        plt.ylim(ymin, ymax)
        plt.xlim(0,min_epoch)
        plt.title(run)
        plt.plot(x, y_OK, x, y_BAD, x, y_prod)
        plt.legend(['F1 OK', 'F1 BAD', 'F1 prod'], loc='lower right')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot learning curves for QE runs.')
    parser.add_argument('log_files', type=str, help='log files', nargs='+')
    args = parser.parse_args()
    main(args)
