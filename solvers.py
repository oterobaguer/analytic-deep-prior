import numpy as np
import tensorflow as tf
from misc import penalty, residual


def tikhonov(A, y_delta, alpha, pen='l2'):
    if pen is 'l1':
        return tikhonov_l1(A, y_delta, alpha)
    if pen is 'l2':
        return tikhonov_l2(A, y_delta, alpha)
    return None


def tikhonov_l2(A, y_delta, alpha):
    W = np.linalg.inv(np.matmul(np.transpose(A), A) + alpha*np.eye(A.shape[0]))
    W = np.matmul(W, np.transpose(A))
    return np.matmul(W, y_delta)


def tikhonov_l1(A, y_delta, alpha):
    return fista(A, np.zeros(shape=y_delta.shape), y_delta, alpha, iter=-1, pen='l1')


def fista(A, x0, y_delta, alpha, lam=None, iter=-1, pen='l2', callback=None, verbose=0):
    tol = 1e-6
    if lam is None:
        lam = 1/(2*np.linalg.norm(A, 2)**2)
    n = x0.shape[0]
    W = np.eye(n) - lam * np.matmul(np.transpose(A), A)
    b = lam * np.matmul(np.transpose(A), y_delta)

    def prox(x, eps, pen):
        if pen is 'l2':
            return x / (1 + eps)
        elif pen is 'l1':
            return np.sign(x) * np.maximum(np.abs(x) - eps, 0.)
        else:
            return x

    xk = x0
    yk = xk
    tk = 1

    k = 0
    while True:
        x0 = xk
        xk = np.matmul(W, yk) + b
        xk = prox(xk, alpha * lam, pen)
        t0 = tk
        tk = (1 + np.sqrt(1+4*tk*tk))/2
        yk = xk + (xk-x0) * (t0-1)/tk
        if callback is not None:
            if verbose and k % 100 == 0:
                print('r=', residual(A, xk, y_delta))
            callback(x=xk, res=residual(A, xk, y_delta), pen=penalty(xk, pen), lam=lam)
        k += 1
        if (iter == -1 and penalty(xk-x0) < tol) or (iter != -1 and k >= iter):
            if verbose:
                print('FISTA ended -- {} used'.format(k))
            break

    return xk


def analytic_deep_prior(A, z, y_delta, alpha, lam=1.0, pen='l2', iters=20000, layers=20, lr=0.01, train_alpha=False, train_lambda=False, warm=False, unroll_fista=False, verbose=1,
                        callback=None, callback_interval=1, basis=None):
    if basis is None:
        basis_tensor = tf.constant(np.eye(A.shape[0]), dtype=tf.float64)
    else:
        basis_tensor = tf.constant(basis, dtype=tf.float64)

    np.random.seed(10)
    x_shape = z.shape
    n = x_shape[0]

    y_tensor = tf.constant(y_delta, dtype=tf.float64)

    A_tensor = tf.constant(A, dtype=tf.float64)
    _B_tensor = tf.Variable(A, dtype=tf.float64)
    B_tensor = tf.matmul(_B_tensor, basis_tensor)
    if train_alpha:
        _alpha_tensor = tf.Variable(np.sqrt(alpha), dtype=tf.float64)
        alpha_tensor = tf.square(_alpha_tensor)
    else:
        alpha_tensor = tf.constant(alpha, dtype=tf.float64)

    if train_lambda:
        _lambda_tensor = tf.Variable(np.log(lam), dtype=tf.float64)
        lambda_tensor = tf.exp(_lambda_tensor)
    else:
        lambda_tensor = tf.constant(lam, dtype=tf.float64)

    z_holder = tf.placeholder(tf.float64, shape=x_shape)

    W = tf.eye(n, dtype=tf.float64) - lambda_tensor * tf.matmul(tf.transpose(B_tensor), B_tensor)
    b = lambda_tensor * tf.matmul(tf.transpose(B_tensor), y_tensor)

    def prox(x):
        eps = alpha_tensor * lambda_tensor
        if pen is 'l1':
            return tf.sign(x) * tf.nn.relu(tf.abs(x) - eps)
        elif pen is 'l2':
            return x / (1 + eps)
        else:
            return x

    x_tensor = z_holder

    if unroll_fista:
        # layers correspond to FISTA iterations
        z_tensor = x_tensor
        tk = 1
        for i in range(layers):
            x_tensor_prev = x_tensor
            x_tensor = tf.matmul(W, z_tensor) + b
            x_tensor = prox(x_tensor)
            t_prev = tk
            tk = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            z_tensor = x_tensor + (x_tensor - x_tensor_prev) * (t_prev - 1) / tk
    else:
        # layers correspond to ISTA iterations
        for i in range(layers):
            x_tensor = tf.matmul(W, x_tensor) + b
            x_tensor = prox(x_tensor)

    residual_tensor = tf.reduce_sum(tf.square(y_tensor - tf.matmul(A_tensor, tf.matmul(basis_tensor, x_tensor))))
    opt = tf.train.GradientDescentOptimizer(lr).minimize(residual_tensor)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = None
    for i in range(iters):
        x, res, lam, alpha, B = sess.run([x_tensor, residual_tensor, lambda_tensor, alpha_tensor, _B_tensor],
                                         {z_holder: z})
        sess.run([opt], {z_holder: z})
        if warm:
            z = x
        if verbose and i % callback_interval == 0:
            # clear_output(wait=True)
            print('iter={}, residual={} '.format(i, res), end="\r")
            if callback is not None:
                callback(x=x, res=res, pen=penalty(x, pen), lam=lam, alpha=alpha, B=B)
    return x
