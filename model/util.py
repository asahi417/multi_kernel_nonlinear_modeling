import numpy as np


def realize_loop(regs, names, n_sig, realize=3, size=7000, noise_std=0.01):
    errors = dict()
    dicts = dict()

    for name, reg in zip(names, regs):
        print("mode:%s" % name)
        __er = 0
        __dict = 0
        for r in range(realize):
            print("realize (%i/%i)" % (r, realize))
            np.random.seed(seed=r)
            x, y, y_noise = n_sig(size, noise_std)
            reg.fit(x, y_noise)
            __er += np.array(reg.r_error)
            __dict += np.array(reg.r_dict_size)
        errors[name] = __er/realize
        dicts[name] = __dict/realize
    _error = []
    _name = []
    _dict = []
    for k in errors.keys():
        _error.append(errors[k])
        _dict.append(dicts[k])
        _name.append(k)
    return _name, _error, _dict


def save_stats(_path, _regression):
    """ save statistics of regression instance """
    np.savez(_path, error=_regression.r_error, dict_size=_regression.r_dict_size,
             coefficient=_regression.co, dictionary=_regression.dict)


def ma(target, window):
    """ Smoothing sequence for visualization """
    return [np.mean(target[_s:_s + window]) for _s in range(len(target) - window)]


def plot_full(error, dictionary, name, window=50, save_path=None, out_legend=True, remove_key=None, target_key=None):
    """ Plot result of multiple result """

    from matplotlib import pyplot as plt

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    plt.figure(0)
    for _tmp, _name in zip(dictionary, name):
        if remove_key is not None and remove_key in _name:
            continue
        if target_key is not None and target_key not in _name:
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
        if remove_key is not None and remove_key in _name:
            continue
        if target_key is not None and target_key not in _name:
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
