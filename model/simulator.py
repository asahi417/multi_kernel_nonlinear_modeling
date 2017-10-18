import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_signal(x, y):
    if x.ndim == 1:
        _df = pd.Series(y, index=x).sort_index()
        _df.plot()
        plt.grid()
    elif x.ndim == 2:
        plt.scatter(_x[:, 0], _x[:, 1], s=1, c=_y, cmap='summer')
    plt.title("Simulated System", fontsize=25)
    plt.xlabel("Input", fontsize=20)
    plt.ylabel("Output", fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()

def nonlinear_siganl_1d(size=10000, noise_std=0.01):
    """
    y = exp(-20(x-0.1)^2)-2exp(-20(x-0.8)^2) + e
    - x in uniform[0, 1]
    

    The 1 dimensional nonlinear system, used in following paper
    - Yukawa, Masahiro, and Ryu-ichiro Ishii. "On adaptivity of online model selection
      method based on multikernel adaptive filtering." APSIPA, 2013 Asia-Pacific. IEEE, 2013.
    """
    _x = np.random.rand(size)
    _y = np.exp(-20*(_x-0.1)**2)-2*np.exp(-20*(_x-0.8)**2)
    _y_noise = _y + np.random.randn(size)*noise_std
    return _x, _y, _y_noise

def nonlinear_siganl_2d(size=10000, noise_std=0.01):
    """

    - x in uniform[-0.5, 0.5]

    The 2 dimensional nonlinear system, used in following paper
    - Toda, Osamu, and Masahiro Yukawa. "On Kernel design for online model selection by
      Gaussian multikernel adaptive filtering." APSIPA. IEEE, 2014.
    """


    _x = np.random.rand(size, 2)-0.5
    _h = np.array([0.4, 0.3, -0.3, 0.2])
    _c = np.array([[0.2, 0.2], [-0.3, -0.1], [-0.1, 0.2], [0, -0.2]])
    _s = np.array([0.05, 0.5, 0.1, 0.2])
    _y = 0
    for __h, __c, __s in zip(_h, _c, _s):
        _tmp = (_x-__c)**2
        _y += __h*np.exp(-_tmp.sum(1)/2/__s)
    _y_noise = _y + np.random.randn(size)*noise_std
    return _x, _y, _y_noise