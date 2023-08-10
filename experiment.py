import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from tqdm.auto import tqdm
import os

import pyproximal as pyprox
import pylops

from numba import njit

@njit
def prox_fx(x: np.ndarray,
            er: np.ndarray, cov: np.ndarray,
            risk_aversion: float=1.0,
            eta: float=1.0,
            benchmark_pf:np.ndarray|None=None,
            current_pf:np.ndarray|None=None,
            reference_pf: np.ndarray|None=None,
            penalty_current_l1: float=0.0,
            penalty_refer_l1: float=0.0,
            penalty_current_l2: float=0.0,
            penalty_refer_l2: float=0.0,
            param_rb: float=0.0,
            rb: np.ndarray|None=None,
            phi: float=1.0,
            tol: int=1e-7, max_iter: int=1e+3) -> np.ndarray:

    assert risk_aversion > 0.
    assert eta >= 0.
    assert param_rb >= 0.
    assert penalty_current_l2 >= 0.
    assert penalty_refer_l2 >= 0.
    assert phi >= 0.
    assert tol > 0.
    assert max_iter > 0.

    benchmark_pf = np.zeros_like(x) if benchmark_pf is None else benchmark_pf
    current_pf = np.zeros_like(x) if current_pf is None else current_pf
    reference_pf = np.zeros_like(x) if reference_pf is None else reference_pf
    rb = np.ones_like(x) if rb is None else rb
    rb = rb / np.sum(rb)

    n = len(x)

    k = 0
    Q = 2 * risk_aversion * cov + penalty_current_l2 + penalty_refer_l2 + phi * np.identity(n)
    R = eta * er + 2 * risk_aversion * cov @ benchmark_pf + penalty_current_l2 * current_pf + penalty_refer_l2 * reference_pf + phi * x
    while True:
        xold = x.copy()
        if param_rb > 0:
            for i in range(n):
                x_first = R[i] - np.sum(x[:i] * Q[i, :i]) - np.sum(xold[i+1:] * Q[i, i+1:])
                x_second = np.sqrt((-1 * x_first) ** 2 + 4 * param_rb * rb[i] * Q[i, i])
                x[i] = (x_first + x_second) / (2*Q[i, i])
        else:
            x = np.linalg.solve(Q, R)

        k += 1
        if np.sum(np.abs(x - xold)) < tol or k >= max_iter:
            break

    return x


def prox_fy(y, phi):
    # phi is redundant
    return pyprox.Simplex(len(y), radius=1.0, engine='numba').prox(y, phi)
    # return y


k = 0
x = np.zeros(12)
y = np.zeros(12)
u = np.zeros(12)
er = pd.read_csv('er.csv', index_col=0).values.ravel()
cov = pd.read_csv('cov.csv', index_col=0).values
phi = 1.0
model = {
    'MeanVar': {
        'risk_aversion': 1.0,
        'eta': 1.0,
        'param_rb': 0.0
    },
    'ERC': {
        'risk_aversion': 0.5,
        'eta': 0.0,
        'param_rb': 5.0
    },
    'RB': {
        'risk_aversion': 0.5,
        'eta': 0.0,
        'param_rb': 1.0,
        'rb': np.array([1.,1.,2.,2.,3.,3.,4.,4.,5.,5.,6.,6.])
    }
}
tol = 1e-10
n_iter = 200
while True:
    xold = x.copy()
    yold = y.copy()
    uold = u.copy()

    # ADMM
    x = prox_fx(x=y - u, er=er, cov=cov, phi=phi, **model['ERC'])
    y = prox_fy(x + u, phi=phi)
    u += x - y

    # ADMM param update (paper p.56)
    r = x - y
    s = phi * (y - yold)
    mu = 1e+1
    tau = 2.0
    error_primal = np.sum(r**2)
    error_dual = np.sum(s**2)
    if error_primal > mu*error_dual:
        phi *= tau
        u /= tau
    elif error_dual > mu*error_primal:
        phi /= tau
        u *= tau

    # print('x: ', x)
    # print('y: ', y)
    # print('u: ', u)

    if max(sum((xold-x)**2),
           sum((yold-y)**2),
           sum((x-y)**2)) <= tol or k >= n_iter:
        print('iter exceeds max_iter') if k >= n_iter else print(f'converged after {k} iterations.')
        break

    k += 1
    print(f'° iter {k}')
    print('\tweights\t\t\t\t', np.round(x, 3)*100)
    print('\trisk contributions\t', np.round((x * (cov @ x)) / (x.T@cov@x), 3)*100)

# x = y
print(sum(x))
# print(prox_fx(er=pd.read_csv('er.csv', index_col=0).values.ravel(),
#         cov=pd.read_csv('cov.csv', index_col=0).values,
#         risk_aversion=1.0,
#         phi=1.0
#         # max_iter=1
#               )
# )