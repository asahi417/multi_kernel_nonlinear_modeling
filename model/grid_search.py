import numpy as np
from itertools import product


def grid_search_full(_regression, _x, _y, name_of_reg="", dictionary_th=[0], alpha_dict=[10**-10], eta=[0.06, 0.07]):
    a = np.arange(1, 10)
    b = np.arange(-4, 2)
    k_p = [i[0] * 10.0 ** i[1] for i in product(a, b)]
    dict_size = 20
    _regressions = []
    _p = []
    full_size = len(eta)*len(dictionary_th)*len(alpha_dict)
    for _n, (_d, _a, _e) in enumerate(product(dictionary_th, alpha_dict, eta)):
        _name = "%s_dict_%s_alpha_%s_eta_%s" % (name_of_reg, str(_d), str(_a), str(_e))
        print("mode:%i/%i (%s)" % (_n, full_size, _name))
        _reg = _regression(eta0=_e, kernel_parameter=k_p, dictionary_th=_d, alpha_dict=_a, dict_size=dict_size)
        _reg.fit(_x, _y)
        _regressions.append(_reg)
        _p.append(_name)
    return _regressions, _p


def grid_search_reg(_regression, _x, _y, dictionary_th=[0], alpha_dict=[10**-10], eta0=0.01):
    a = np.arange(1, 10)
    b = np.arange(-4, 2)
    kernel_parameter = [i[0] * 10.0 ** i[1] for i in product(a, b)]
    dict_size = 20
    _regressions = []
    _p = []
    full_size = len(dictionary_th)*len(alpha_dict)
    for _n, (_d, _a) in enumerate(product(dictionary_th, alpha_dict)):
        print("mode:%i/%i" % (_n, full_size))
        _reg = _regression(eta0=eta0, kernel_parameter=kernel_parameter, dictionary_th=_d,
                           dict_size=dict_size, alpha_dict=_a)
        _reg.fit(_x, _y)
        _regressions.append(_reg)
        _name = "dict_%s_a_%s" % (str(_d), str(_a))
        _p.append(_name)
    return _regressions, _p


def grid_search_eta(_regression, _x, _y, dictionary_th=0, alpha_dict=10**-10, eta=[0.06, 0.07]):
    a = np.arange(1, 10)
    b = np.arange(-4, 2)
    kernel_parameter = [i[0] * 10.0 ** i[1] for i in product(a, b)]
    dict_size = 20
    _regressions = []
    _p = []
    full_size = len(eta)
    for _n, eta0 in enumerate(eta):
        print("mode:%i/%i" % (_n, full_size))
        _reg = _regression(eta0=eta0, kernel_parameter=kernel_parameter, dictionary_th=dictionary_th,
                           dict_size=dict_size, alpha_dict=alpha_dict)
        _reg.fit(_x, _y)
        _regressions.append(_reg)
        _name = "dict_%s_a_%s_eta_%s" % (str(dictionary_th), str(alpha_dict), eta0)
        _p.append(_name)
    return _regressions, _p
