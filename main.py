import numpy as np
import pandas as pd
from ADMM_prototype import *
import ADMM_prototype


def main():
    er = pd.read_csv('er.csv', index_col=0).values.ravel()
    cov = pd.read_csv('cov.csv', index_col=0).values
    model = 'ERC'
    params_model = {
        'MVO': {
            'fx': {
                'risk_aversion': 1.0,
                'eta': 1.0,
                'param_rb': 0.0,
            },
            'fy': {

            }
        },
        'ERC': {
            'fx': {
                'risk_aversion': 1.0,
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
            'fx': {
                'risk_aversion': 0.5,
                'eta': 0.0,
                'param_rb': 1.0,
                'rb': np.array([1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.]),
            },
            'fy': {

            }
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
             'Upper': [0.2]},
        'Turnover limit': None,
        # 'Transaction costs limit': None,
        'Leverage limit': None,
        'Long/short exposure': None,
        'Benchmarking': None,
        'Tracking error floor': None,
        'Active share floor': None,
        #     'Num of active bets': None
    }
    tol = 1e-8
    max_iter = 1000
    x = solve(er=er, cov=cov,
              model=model, params_model=params_model,
              constraints=constraints,
              tol=tol, max_iter=max_iter)

    print(f'{model} Portfolio')
    print('Final weights\t\t\t', np.round(x, 4) * 100)
    print('Final risk contributions', np.round((x * (cov @ x)) / (x.T @ cov @ x), 4) * 100)
    print(sum(x))

main()
