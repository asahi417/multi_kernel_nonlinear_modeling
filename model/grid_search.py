import numpy as np
from itertools import product


def grid_search(_regression, _x, _y, dictionary_th=[0], alpha_dict=[10**-10]):
    a = np.arange(1, 10)
    b = np.arange(-4, 2)
    kernel_parameter = [i[0] * 10.0 ** i[1] for i in product(a, b)]
    eta0 = 0.01
    dict_size = 20
    _regressions = []
    _p = []
    full_size = len(dictionary_th)*len(alpha_dict)
    for _n, (_d, _a) in enumerate(product(dictionary_th, alpha_dict)):
        print("mode:%i/%i" % (_n, full_size))
        _reg = _regression(
            eta0=eta0,
            kernel_parameter=kernel_parameter,
            dictionary_th=_d,
            dict_size=dict_size,
            alpha_dict=_a
        )
        _reg.fit(_x, _y)
        _regressions.append(_reg)
        _name = "dict_%s_a_%s" % (str(_d), str(_a))
        _p.append(_name)
    return _regressions, _p
