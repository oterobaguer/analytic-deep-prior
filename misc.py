import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter


class Callback:
    def __init__(self, x_true=None, pen='l2'):
        self.x_true = x_true
        self.x_seq = []
        self.res_seq = []
        self.true_err_seq = []
        self.pen_seq = []
        self.alpha_seq = []
        self.lam_seq = []
        self.B_seq = []
        self.pen = pen

    def __call__(self, *args, **kwargs):
        if 'res' in kwargs:
            self.res_seq += [kwargs['res']]
        if 'pen' in kwargs:
            self.pen_seq += [kwargs['pen']]
        if 'alpha' in kwargs:
            self.alpha_seq += [kwargs['alpha']]
        if 'lam' in kwargs:
            self.lam_seq += [kwargs['lam']]
        if self.x_true is not None and 'x' in kwargs:
            self.true_err_seq += [penalty(self.x_true - kwargs['x'], self.pen)]
        if 'x' in kwargs:
            self.x_seq += [kwargs['x']]
        if 'B' in kwargs:
            self.B_seq += [kwargs['B']]


def get_operator(name, n=100):
    h = 1 / n
    A = None
    if name is 'integral':
        A = h * (np.tril(np.ones(n)) - (1 / 2) * np.diag(np.ones(n)))
    elif name is 'smoothing':
        A = gaussian_filter(np.eye(n, n), 5)

    L = np.linalg.norm(A, 2) ** 2
    pts = np.arange(h / 2, 1, h)
    return A, L, pts


def penalty(x, pen='l2'):
    if pen is 'l2':
        return np.dot(x.reshape(-1), x.reshape(-1))
    elif pen is 'l1':
        return np.linalg.norm(x, 1)
    else:
        return 0


def residual(A, x, y_delta):
    r = np.matmul(A, x) - y_delta
    return penalty(r)


def curvature(x, y):
    c = curvature_splines(np.array(x), np.array(y))
    return c


def curvature_1D(y):
    c = curvature_splines_1D(np.array(y))
    return c


def curvature_splines(x, y):
    t = np.arange(x.shape[0])
    fx = UnivariateSpline(t, y, k=5, s=0.1)
    fy = UnivariateSpline(t, y, k=5, s=0.1)
    x1 = fx.derivative(1)(t)
    x11 = fx.derivative(2)(t)
    y1 = fy.derivative(1)(t)
    y11 = fy.derivative(2)(t)
    return (y1 * x11 - x1 * y11) / np.power(x1 ** 2 + y1 ** 2, 3 / 2)


def curvature_splines_1D(y):
    t = np.arange(y.shape[0])
    fy = UnivariateSpline(t, y, k=5, s=1.0)
    y1 = fy.derivative(1)(t)
    y11 = fy.derivative(2)(t)
    return y11 / np.power(1 + y1 ** 2, 3 / 2)
