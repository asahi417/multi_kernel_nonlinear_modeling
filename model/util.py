import numpy as np
from matplotlib import pyplot as plt


def save_stats(_path, _regression):
    """ save statistics of regression instance """
    np.savez(_path, error=_regression.r_error, dict_size=_regression.r_dict_size,
             coefficient=_regression.co, dictionary=_regression.dict)


def ma(target, window):
    """ Smoothing sequence for visualization """
    return [np.mean(target[_s:_s + window]) for _s in range(len(target) - window)]


def plot_full(error, dictionary, name, window=50, save_path=None, out_legend=True, remove_key=None):
    """ Plot result of multiple result """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    plt.figure(0)
    for _tmp, _name in zip(dictionary, name):
        if remove_key in _name:
            continue
        plt.plot(_tmp, label=_name)
    if out_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=14)
    else:
        plt.legend(loc="upper left")
    if save_path is not None:
        plt.savefig(save_path+"_dict.eps", bbox_inches="tight")
    plt.show()

    plt.figure(1)
    for _tmp, _name in zip(error, name):
        if remove_key in _name:
            continue
        plt.plot(ma(_tmp, window), label=_name)
        plt.yscale("log")
    if out_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=14)
    else:
        plt.legend(loc="upper left")
    if save_path is not None:
        plt.savefig(save_path+"_loss.eps", bbox_inches="tight")
    plt.show()
