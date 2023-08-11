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
def _ccd_fx(x: np.ndarray,
            er: np.ndarray, cov: np.ndarray,
            risk_aversion: float = 1.0,
            eta: float = 1.0,
            benchmark_pf: np.ndarray | None = None,
            current_pf: np.ndarray | None = None,
            reference_pf: np.ndarray | None = None,
            penalty_current_l2: float = 0.0,
            penalty_refer_l2: float = 0.0,
            param_rb: float = 0.0,
            rb: np.ndarray | None = None,
            phi: float = 1.0,
            tol: int = 1e-7, max_iter: int = 1e+3):
    k = 0
    n = len(x)
    Q = 2 * risk_aversion * cov + (penalty_current_l2 + penalty_refer_l2 + phi) * np.identity(n)
    R = eta * er + 2 * risk_aversion * cov @ benchmark_pf + penalty_current_l2 * current_pf + penalty_refer_l2 * reference_pf + phi * x
    while True:
        xold = x.copy()
        if param_rb > 0:
            for i in range(n):
                x_first = R[i] - np.sum(x[:i] * Q[i, :i]) - np.sum(xold[i + 1:] * Q[i, i + 1:])
                x_second = np.sqrt((-1 * x_first) ** 2 + 4 * param_rb * rb[i] * Q[i, i])
                x[i] = (x_first + x_second) / (2 * Q[i, i])
        else:
            x = np.linalg.solve(Q, R)

        k += 1
        if np.sum(np.abs(x - xold)) < tol or k >= max_iter:
            break

    return x


def prox_fx(x: np.ndarray,
            er: np.ndarray, cov: np.ndarray,
            risk_aversion: float = 1.0,
            eta: float = 1.0,
            benchmark_pf: np.ndarray | None = None,
            current_pf: np.ndarray | None = None,
            reference_pf: np.ndarray | None = None,
            penalty_current_l2: float = 0.0,
            penalty_refer_l2: float = 0.0,
            param_rb: float = 0.0,
            rb: np.ndarray | None = None,
            phi: float = 1.0,
            tol: int = 1e-7, max_iter: int = 1e+3) -> np.ndarray:
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

    x = _ccd_fx(x,
               er, cov,
               risk_aversion, eta,
               benchmark_pf, current_pf, reference_pf,
               penalty_current_l2, penalty_refer_l2,
               param_rb, rb,
               phi, tol, max_iter)

    # normalizing required
    return x / np.sum(x)


def _prox_l1_penalty(y: np.ndarray,
                     sigma: float, g: np.ndarray,
                     phi: float = 1.0) -> np.ndarray:
    return pyprox.L1(sigma=sigma, g=g).prox(y, phi)


def _prox_simplex(y: np.ndarray,
                  budget: float = 1.0,
                  activated: bool = True) -> np.ndarray:
    if activated:
        return pyprox.Simplex(len(y), radius=budget, engine='numba').prox(y, 1.0)
    else:
        return y


def _prox_box(y: np.ndarray,
              lower: float | np.ndarray = -np.inf, upper: float | np.ndarray = np.inf,
              activated: bool = False, phi: float = 1.0) -> np.ndarray:
    if activated:
        return pyprox.Box(lower=lower, upper=upper).prox(y, phi)
    else:
        return y


def _prox_halfspaces():
    # Dykstra's algorithm
    pass


def _prox_l1_ball():
    pass


def _prox_l2_ball():
    pass


def _dykstra(v: np.ndarray, func_list: list | tuple):
    k = 0
    v_list = np.zeros(
        shape=[len(func_list) + 1, len(v)])  # +1 is for v_0 and z_0 (but z_0 is redundant, just for consistency)
    z_list = np.zeros(shape=[len(func_list) + 1, len(v)])  # z_list[0] not used.
    v_list[-1] = v
    while True:
        vold_list = v_list.copy()
        zold_list = z_list.copy()
        v_list[0] = vold_list[-1]  # v_0 <- vold_m
        for i in range(len(v_list) - 1):  # i: 0 ~ m-1
            v_list[i + 1] = func_list[i](v_list[i] + zold_list[i + 1])
            z_list[i + 1] = v_list[i] + zold_list[i + 1] - v_list[i + 1]

        if np.sum((v_list[-1] - vold_list[0]) ** 2) <= tol or k >= max_iter:
            break

        k += 1

    return v_list[-1]


def prox_fy(v: np.ndarray,
            current_pf: np.ndarray | None = None,
            reference_pf: np.ndarray | None = None,
            penalty_current_l1: float = 0.0,
            penalty_refer_l1: float = 0.0,
            const: dict | None = None,
            phi: float = 1.0,
            tol: float = 1e-8, max_iter: float = 200) -> np.ndarray:
    const_default = {
        'Budget & Long only': True,
        'Weight Bounds': None,
    #     'Industry limit': None,
    #     'Group limit': None,
    #     'Sector limit': None,
    #     'Country limit': None,
    #     'Turnover limit': None,
    #     'Transaction costs limit': None,
    #     'Leverage limit': None,
    #     'Long/short exposure': None,
    #     'Benchmarking': None,
    #     'Tracking error floor': None,
    #     'Active share floor': None,
    #     'Num of active bets': None
    }

    if const is not None:
        for key in const_default.keys():
            const[key] = const_default[key] if key not in const.keys() else const[key]
    else:
        const = const_default

    _prox = [
        lambda x: _prox_l1_penalty(x, penalty_current_l1, current_pf, phi),
        lambda x: _prox_l1_penalty(x, penalty_refer_l1, reference_pf, phi),
        lambda x: _prox_simplex(x, True) if const['Budget & Long only'] else _prox_simplex(x, False),
        lambda x: (_prox_box(x, const['Weight Bounds']['Lower'], const['Weight Bounds']['Upper'], True)
                   if const['Weight Bounds'] is not None else _prox_box(x, False)),
    ]

    v = _dykstra(v, _prox)

    return v




k = 0
x = np.zeros(12)
y = np.zeros(12)
u = np.zeros(12)
er = pd.read_csv('er.csv', index_col=0).values.ravel()
cov = pd.read_csv('cov.csv', index_col=0).values
phi = 1.0
params_model = {
    'MeanVar': {
        'risk_aversion': 1.0,
        'eta': 1.0,
        'param_rb': 0.0
    },
    'ERC': {
        'fx': {
            'risk_aversion': 0.5,
            'eta': 0.0,
            'param_rb': 1.0,
            # 'penalty_refer_l2': 0.0,
            # 'reference_pf': np.ones(12)/12
        },
        'fy': {
            # 'penalty_refer_l1': 0.0,
            # 'reference_pf': np.ones(12)/12
        }
    },
    'RB': {
        'risk_aversion': 0.5,
        'eta': 0.0,
        'param_rb': 1.0,
        'rb': np.array([1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.])
    }
}
tol = 1e-10
max_iter = 200
while True:
    xold = x.copy()
    yold = y.copy()
    uold = u.copy()

    # ADMM
    model = 'ERC'
    x = prox_fx(x=y - u, er=er, cov=cov, phi=phi, **params_model[model]['fx'])
    y = prox_fy(x + u, phi=phi, **params_model[model]['fy'])
    u += x - y

    # ADMM param update (paper p.56)
    r = x - y
    s = phi * (y - yold)
    mu = 1e+1
    tau = 2.0
    error_primal = np.sum(r ** 2)
    error_dual = np.sum(s ** 2)
    if error_primal > mu * error_dual:
        phi *= tau
        u /= tau
    elif error_dual > mu * error_primal:
        phi /= tau
        u *= tau

    if max(sum((xold - x) ** 2),
           sum((yold - y) ** 2),
           sum((x - y) ** 2)) <= tol or k >= max_iter:
        print('iter exceeds max_iter') if k >= max_iter else print(f'converged after {k} iterations.')
        break

    k += 1
    print(f'° iter {k}')
    print('\tweights\t\t\t\t', np.round(x, 3) * 100)
    print('\trisk contributions\t', np.round((x * (cov @ x)) / (x.T @ cov @ x), 3) * 100)

print(sum(x))
