import numpy as np
from progressbar import ProgressBar

from . import kernel



import numpy as np
from progressbar import ProgressBar

from model import kernel



class MultiKernelRegressor:

    def __init__(self, eta0=0.3, alpha_element=10**-4, alpha_dict=10**-4, eps=10**-5, dictionary_th=10**-5,
                 dict_size=10, kernel_type="gauss_norm", kernel_parameter=[0.01, 0.1]):
        self.eta0 = eta0
        self.alpha_dict = alpha_dict
        self.alpha_element = alpha_element
        self.eps = eps
        self.dict_th = dictionary_th
        self.dict_size = dict_size
        self.kernel_type = kernel_type
        self.kernel_p = kernel_parameter
        if kernel_type == "gauss_norm":
            self.kernel = kernel.gauss_norm
        elif kernel_type == "gauss":
            self.kernel = kernel.gauss

    def fit(self, x, y):
        """

        double regularizer
        - dictionary point wise regularizer
        - element wise regularizer

         Parameters
        ---------------------------------------
        x: input, shape (full size, feature)
        y: output, shape (full size)
        dict: (dictionary point, feature)
        co: coefficient (kernel number, dictionary point)
        """

        # save variables
        self.r_dict_size = []
        self.r_error = []

        self.dict = np.empty((0, x.shape[1]))
        self.co =  np.empty((len(self.kernel_p), 0))

        p = ProgressBar(maxval=len(y))
        _dict_norm = np.empty(0)
        gamma = 2/(2/(self.eta0+self.eps)-1)

        for ind, (_x, _y) in enumerate(zip(x, y)):
            # add new mesurement to dictionary and coefficient
            self.dict = np.vstack([self.dict, _x])
            _dict_norm = np.hstack([_dict_norm, 0])  # shape (dictionary point, )
            self.co = np.hstack([self.co, np.zeros((len(self.kernel_p), 1))])
            assert len(self.dict) == self.co.shape[1]

            # update coefficient
            _k = np.array([self.kernel(self.dict, _x, _p) for _p in self.kernel_p]) # shape (kernel number, dictionary point)
            assert _k.shape == self.co.shape
            prediction = (_k*self.co).sum()
            error = prediction - _y
            # gradient of loss
            _loss = error*_k/(np.trace(_k.dot(_k.T))+self.eps)
            assert len(self.dict) == self.co.shape[1]
            # gradient of reg
            _reg = 1 - gamma*self.alpha_dict/(_dict_norm+self.eps)  # shape (dictionary point, )
            assert len(_reg) == len(self.dict)
            _reg = self.co*_reg*(_reg >  0)
            assert _reg.shape == self.co.shape
            _reg = (self.co-_reg)/gamma
            # update
            self.co += - self.eta0 * (_loss + _reg)
            # prox
            _reg = 1-self.eta0*self.alpha_element/(np.abs(self.co)+self.eps)  # shape (kernel number, dictionary point)
            # _reg[_reg < 0] = 0
            assert self.co.shape == _reg.shape
            self.co = self.co*_reg*(_reg > 0)

            # refine dictionary
            _dict_norm = (self.co*self.co).sum(0)  # shape (dictionary point, )
            if len(self.dict) >= self.dict_size:
                _index = (_dict_norm > self.dict_th)
                self.dict = self.dict[_index]
                self.co = self.co.T[_index].T
                _dict_norm = _dict_norm[_index]

            # record variables
            self.r_dict_size.append(len(self.dict))
            self.r_error.append(error**2)

            # progressbar
            p.update(ind+1)

if __name__ == '__main__':
    mm = MultiKernelRegressor()
    x = np.ones((100, 3))
    y = np.arange(100)
    mm.fit(x, y)
