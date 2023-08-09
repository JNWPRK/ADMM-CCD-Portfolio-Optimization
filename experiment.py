import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from tqdm.auto import tqdm
import os

import pyproximal as pyprox
import pylops

def prox_fx(x0: np.ndarray,
            er: np.ndarray, cov: np.ndarray,
            risk_aversion: float=1.0,
            benchmark_pf:np.ndarray|None=None,
            current_pf:np.ndarray|None=None,
            reference_pf: np.ndarray|None=None,
            penalty_current_l1: float=0.0,
            penalty_refer_l1: float=0.0,
            penalty_current_l2: float=0.0,
            penalty_refer_l2: float=0.0,
            param_rb: float=0.0,
            rb: np.ndarray|None=None,
            tau: float=1.0,
            tol: int=1e-7, max_iter: int=1e+3) -> np.ndarray:

    assert risk_aversion > 0
    assert param_rb >= 0
    assert penalty_current_l2 >= 0
    assert penalty_refer_l2 >= 0
    assert tau >= 0
    assert tol > 0
    assert max_iter > 0

    benchmark_pf = np.zeros_like(er) if benchmark_pf is None else benchmark_pf
    current_pf = np.zeros_like(er) if current_pf is None else current_pf
    reference_pf = np.zeros_like(er) if reference_pf is None else reference_pf
    rb = np.ones_like(er) if rb is None else rb
    rb = rb / np.sum(rb)

    n = len(x0)

    # if param_rb == 0:
    #     Q = 2 * risk_aversion * cov + penalty_current_l2 + penalty_refer_l2 + tau * np.identity(n)
    #     R = er + 2*risk_aversion*cov@benchmark_pf + penalty_current_l2*current_pf + penalty_refer_l2*reference_pf  + tau*x
    #     x = np.linalg.inv(Q)@R
    #
    k = 0
    x = x0
    Q = 2 * risk_aversion * cov + penalty_current_l2 + penalty_refer_l2 + tau * np.identity(n)
    while True:
        xold = x.copy()
        R = er + 2 * risk_aversion * cov@benchmark_pf + penalty_current_l2 * current_pf + penalty_refer_l2 * reference_pf + tau * xold
        if param_rb > 0:
            lamb = param_rb * rb
            for i in range(n):
                x_first = R[i] - x@Q[i] + x[i]*Q[i, i]
                x_second = np.sqrt((-1*x_first)**2 + 4*lamb[i]*Q[i, i])
                x[i] = (x_first + x_second) / (2*Q[i, i])
        else:
            x = np.linalg.inv(Q)@R

        k += 1
        if np.sum(np.abs(x - xold)) < tol or k >= max_iter:
            break

    return x


def prox_fy(y0):
    return pyprox.projection.SimplexProj(len(y0), 1.0)(y0)


k = 0
x = np.ones(12)
y = np.zeros(12)
u = np.zeros(12)
er = pd.read_csv('er.csv', index_col=0).values.ravel()
cov = pd.read_csv('cov.csv', index_col=0).values
while True:
    xold = x.copy()
    yold = y.copy()
    uold = u.copy()

    x = prox_fx(x0=y - u, er=er, cov=cov, risk_aversion=1.0, param_rb=1.0, tau=1.0)
    y = prox_fy(x + u)
    u = u + x - y
    # print('x', x, sep='\n')
    # print('y', y, sep='\n')
    print('x-y', x-y, sep='\n')
    # print('u', u, sep='\n')

    if all((np.sum(np.abs(xold-x)) < 1e-7,
            np.sum(np.abs(yold-y)) < 1e-7,
            np.sum(np.abs(uold-u)) < 1e-7,)) or k >= 1000:
        print('iter exceeds max_iter') if k >= 1000 else print('converged.')
        break

    k += 1

# x = y
print(np.round(x, 3)*100)
print(np.round(x * (cov @ x) / (x.T@cov@x), 3)*100)
print(sum(x))
# print(prox_fx(er=pd.read_csv('er.csv', index_col=0).values.ravel(),
#         cov=pd.read_csv('cov.csv', index_col=0).values,
#         risk_aversion=1.0,
#         tau=1.0
#         # max_iter=1
#               )
# )