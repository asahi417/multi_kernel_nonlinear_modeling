
import numpy as np
from glob import glob
from itertools import product
from model import nonlinear_signal_1d as n_sig
from model import MultiKernelRegressionFBS01 as Regressor_fbs
from model import MultiKernelRegressionPDA01 as Regressor_pda
from model.util import save_stats, realize_loop

order ="1"


eta = 0.3
dictionary_th = 10**-10
pda_alpha = 10**-6
fbs_alpha = 10**-8

regs = []
names = []
a = np.arange(1, 10)
b = np.arange(-4, 2)
kernel_parameter = [i[0] * 10.0 ** i[1] for i in product(a, b)]
dict_size = 20

for name, reg in zip(["fbs", "pda"], [Regressor_fbs, Regressor_pda]):
    if name == "fbs":
        alpha = fbs_alpha
    else:
        alpha = pda_alpha
    _name = "%s_dict_%s_a_%s_eta_%s" % (name, str(dictionary_th), str(alpha), str(eta))
    names.append(_name)
    _reg = reg(eta0=eta, kernel_parameter=kernel_parameter, dictionary_th=dictionary_th,
                   dict_size=dict_size, alpha_dict=alpha)
    regs.append(_reg)

realize = 100
name, error, dictionary = realize_loop(regs, names, n_sig, realize, size=10000, noise_std=0.01)
np.savez("./results/%sd/realize_%i.npz" % (order, realize), name=name, error=error, dictionary=dictionary,
         param_alpha=[fbs_alpha, pda_alpha], param_dictionary_th=dictionary_th, param_eta=eta)