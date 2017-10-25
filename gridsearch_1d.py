import numpy as np
from glob import glob
from model import grid_search_full
from model import nonlinear_signal_1d as n_sig
from model import MultiKernelRegressionFBS01 as Regressor_fbs
from model import MultiKernelRegressionPDA01 as Regressor_pda
from model.util import save_stats


# order 1 for 1d signal and 2 for 2d signal
order = "1"

dictionary_th = [10**-10]
pda_alpha = [10**-12, 10**-10, 10**-8, 10**-6]
fbs_alpha = [10**-12, 10**-10, 10**-8, 10**-6]

pda_eta = [0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]
fbs_eta = [0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]


size = 10000
noise_std = 0.01
x, y, y_noise = n_sig(size, noise_std)

for _name, _regressor in zip(["fbs", "pda"], [Regressor_fbs, Regressor_pda]):
    print(_name)
    if _name == "fbs":
        alpha = fbs_alpha
        eta = fbs_eta
    elif _name == "pda":
        alpha = pda_alpha
        eta = pda_eta

    re = grid_search_full(_regressor, x, y_noise, _name, dictionary_th=dictionary_th, alpha_dict=alpha, eta=eta)
    # save
    for __reg, __name in zip(re[0], re[1]):
        save_stats("./results/%sd/%s.npz" % (order, __name), __reg)