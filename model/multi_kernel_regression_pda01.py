import numpy as np
from progressbar import ProgressBar
from . import kernel


class MultiKernelRegressionPDA01:

    r_dict_size = None
    r_error = None
    dict = None
    co = None

    def __init__(self, eta0=0.3, alpha_dict=10**-4, eps=10**-5, dictionary_th=10**-5,
                 dict_size=10, kernel_type="gauss_norm", kernel_parameter=[0.01, 0.1]):
        self.eta0 = eta0
        self.alpha_dict = alpha_dict
        self.eps = eps
        self.dict_th = dictionary_th
        self.dict_size = dict_size
        self.kernel_type = kernel_type
        self.kernel_p = kernel_parameter
        if kernel_type == "gauss_norm":
            self.kernel = kernel.gauss_norm
        elif kernel_type == "gauss":
            self.kernel = kernel.gauss

    def fit(self, __x, __y):
        """

        double regularize
        - dictionary point wise regularize
        - element wise regularize

         Parameters
        ---------------------------------------
        __x: input, shape (full size, feature)
        __y: output, shape (full size)
        dict: (dictionary point, feature)
        co: coefficient (kernel number, dictionary point)
        """

        # save variables
        self.r_dict_size = []
        self.r_error = []

        self.dict = np.empty((0, __x.shape[1]))
        self.co = np.empty((len(self.kernel_p), 0))

        p = ProgressBar(maxval=len(__y))
        _dict_norm = np.empty(0)

        sum_grad = np.empty((len(self.kernel_p), 0))

        for ind, (_x, _y) in enumerate(zip(__x, __y)):
            # add new input to dictionary and coefficient
            self.dict = np.vstack([self.dict, _x])
            _dict_norm = np.hstack([_dict_norm, 0])  # shape (dictionary point, )
            self.co = np.hstack([self.co, np.zeros((len(self.kernel_p), 1))])
            sum_grad = np.hstack([sum_grad, np.zeros((len(self.kernel_p), 1))])
            # update coefficient
            # shape (kernel number, dictionary point)
            _k = np.array([self.kernel(self.dict, _x, _p) for _p in self.kernel_p])
            prediction = (_k*self.co).sum()
            error = prediction - _y
            # gradient of loss
            sum_grad += error*_k/(np.trace(_k.dot(_k.T))+self.eps)
            _reg = 1-self.alpha_dict/(np.abs(_dict_norm)+self.eps)  # shape (kernel number, dictionary point)
            self.co = - self.eta0*sum_grad*_reg*(_reg > 0)

            # refine dictionary
            _dict_norm = (self.co*self.co).sum(0)  # shape (dictionary point, )
            if len(self.dict) >= self.dict_size:
                _index = (_dict_norm > self.dict_th)
                self.dict = self.dict[_index]
                self.co = self.co.T[_index].T
                _dict_norm = _dict_norm[_index]
                sum_grad = sum_grad.T[_index].T

            # record variables
            self.r_dict_size.append(len(self.dict))
            self.r_error.append(error**2)

            # progressbar
            p.update(ind+1)


if __name__ == '__main__':
    mm = MultiKernelRegressionPDA01()
    x = np.ones((100, 3))
    y = np.arange(100)
    mm.fit(x, y)
