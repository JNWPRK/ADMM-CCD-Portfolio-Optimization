import numpy as np
import pandas as pd
from ADMM_prototype import *


def main():
    k = 0
    x = np.zeros(12)
    y = np.zeros(12)
    u = np.zeros(12)
    er = pd.read_csv('er.csv', index_col=0).values.ravel()
    cov = pd.read_csv('cov.csv', index_col=0).values
    phi = 1.0
    params_model = {
        'MVO': {
            'fx': {
                'risk_aversion': 1.0,
                'eta': 1.0,
                'param_rb': 0.0
            },
            'fy': {

            }
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
    constraints = {
        'Budget': 1.0,
        'Weight bounds': {
            'Lower': 0.0,
            'Upper': 1.0,
        },
        'Industry limit': None,
        'Group limit': None,
        'Sector limit': None,
        'Country limit':
            # None,
            {'Lower': [0.0],
            'Upper': [0.24]},
        'Turnover limit': None,
        # 'Transaction costs limit': None,
        'Leverage limit': None,
        'Long/short exposure': None,
        'Benchmarking': None,
        'Tracking error floor': None,
        'Active share floor': None,
        #     'Num of active bets': None
    }
    model = 'ERC'
    tol = 1e-8
    max_iter = 1000
    while True:
        xold = x.copy()
        yold = y.copy()

        # ADMM
        x = prox_fx(x=y - u, er=er, cov=cov, phi=phi, **params_model[model]['fx'])
        y = prox_fy(x + u, phi=phi, **params_model[model]['fy'], const=constraints)
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
            print('ADMM: Iter exceeds max_iter. The solution may be incorrect.') if k >= max_iter else print(f'ADMM: Converged after {k} iterations.')
            break


        k += 1
        verbose_cycle = 5
        if k % verbose_cycle == 0:
            print(f'° iter {k}')
            print('\tweights\t\t\t\t', np.round(x, 4) * 100)
            print('\trisk contributions\t', np.round((x * (cov @ x)) / (x.T @ cov @ x), 4) * 100)

    # x /= np.sum(x)
    print(f'{model} Portfolio')
    print(sum(x))


main()
