"""
1 dimensional nonlinear system
- y = exp(-20(x-0.1)^2)-2exp(-20(x-0.8)^2) + e
- x in uniform[0, 1]
- reference:
  Yukawa, Masahiro, and Ryu-ichiro Ishii. "On adaptivity of online model selection
  method based on multikernel adaptive filtering." APSIPA, 2013 Asia-Pacific. IEEE, 2013.

2 dimensional nonlinear system
- x in uniform[-0.5, 0.5]
- reference:
  Toda, Osamu, and Masahiro Yukawa. "On Kernel design for online model selection by
  Gaussian multikernel adaptive filtering." APSIPA. IEEE, 2014.
"""

import numpy as np
import pandas as pd


def plot_signal_1d(n=1000, save_path=None):
    from matplotlib import pyplot as plt

    x, y, n = nonlinear_signal_1d(n)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    _df = pd.Series(y, index=x.flatten()).sort_index()
    _df.plot()
    plt.grid()
    plt.xlabel("Input", fontsize=20)
    plt.ylabel("Output", fontsize=20)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def nonlinear_signal_1d(size=10000, noise_std=0.01):
    _x = np.random.rand(size)
    _y = np.exp(-20 * (_x - 0.1) ** 2) - 2 * np.exp(-20 * (_x - 0.8) ** 2)
    _y_noise = _y + np.random.randn(size) * noise_std
    _x = _x.reshape(-1, 1)
    return _x, _y, _y_noise


def plot_signal_2d(n=1000, c_n=50, domain=[-1, 1], save_path=None):
    from matplotlib import pyplot as plt

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15

    _x1 = np.linspace(domain[0], domain[1], n)
    _x2 = np.linspace(domain[0], domain[1], n)
    _x1, _x2 = np.meshgrid(_x1, _x2)
    _x = np.array([_x1.reshape(n ** 2), _x2.reshape(n ** 2)]).T
    _h = np.array([0.4, 0.3, -0.3, 0.2])
    _c = np.array([[0.2, 0.2], [-0.3, -0.1], [-0.1, 0.2], [0, -0.2]])
    _s = np.array([0.05, 0.5, 0.1, 0.2])
    _y = 0
    for __h, __c, __s in zip(_h, _c, _s):
        _tmp = (_x - __c) ** 2
        _y += __h * np.exp(-_tmp.sum(1) / 2 / __s)
    _y = _y.reshape(n, n)

    plt.contour(_x1, _x2, _y, c_n)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.xlabel("In 1", fontsize=20)
    plt.ylabel("In 2", fontsize=20)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def nonlinear_signal_2d(size=10000, noise_std=0.01):
    _x = np.random.rand(size, 2) - 0.5
    _h = np.array([0.4, 0.3, -0.3, 0.2])
    _c = np.array([[0.2, 0.2], [-0.3, -0.1], [-0.1, 0.2], [0, -0.2]])
    _s = np.array([0.05, 0.5, 0.1, 0.2])
    _y = 0
    for __h, __c, __s in zip(_h, _c, _s):
        _tmp = (_x - __c) ** 2
        _y += __h * np.exp(-_tmp.sum(1) / 2 / __s)
    _y_noise = _y + np.random.randn(size) * noise_std
    return _x, _y, _y_noise
