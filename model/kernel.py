import numpy as np

"""
_x: dictionary point, shape (data number, feature) or (feature,)
_y: input, shape (feature,)
"""

def gauss_norm(_x, _y, _s):
    """
    Normalized RBF kernel
    _s: variance of gauss
    """
    distance = ((_x - _y)**2).sum(1)
    assert len(distance) == len(_x)
    return np.exp(-distance/2/_s)/(2*np.pi*_s)**(len(_y)/2)

def gauss(_x, _y, _s):
    """
    RBF kernel
    _s: variance of gauss
    """
    distance = ((_x - _y)**2).sum(1)
    assert len(distance) == len(_x)
    return np.exp(-distance/2/_s)


# def linear(_x, _y):
#     assert _x.shape[1] == len(_y)
#     return ((_x - _y)**2).sum(1)

if __name__ == '__main__':
    # 2 dictionary and 3 feature
    a = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [-10.1, -10.2, -10.3]])
    b = np.array([0.1, 0.2, 0.3])
    print(gauss(a, b, 10))
    print(gauss(a, b, 1))
    print(gauss(a, b, 0.1))

    print(gauss_norm(a, b, 10))
    print(gauss_norm(a, b, 1))
    print(gauss_norm(a, b, 0.1))
