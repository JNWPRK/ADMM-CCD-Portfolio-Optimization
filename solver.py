import numpy as np
import pyproximal as pyprox
import pylops
from numba import njit
from scipy.optimize import bisect


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
            tol: int = 1e-7, max_iter: int = 200) -> np.ndarray:
    k = 0
    n = len(x)
    Q = 2 * risk_aversion * cov + (penalty_current_l2 + penalty_refer_l2 + phi) * np.identity(n)
    R = eta * er + 2 * risk_aversion * cov @ benchmark_pf + penalty_current_l2 * current_pf + penalty_refer_l2 * reference_pf + phi * x

    if param_rb > 0:
        while True:
            xold = x.copy()
            for i in range(n):
                x_first = R[i] - np.sum(x[:i] * Q[i, :i]) - np.sum(xold[i + 1:] * Q[i, i + 1:])
                x_second = np.sqrt((-1 * x_first) ** 2 + 4 * param_rb * rb[i] * Q[i, i])
                x[i] = (x_first + x_second) / (2 * Q[i, i])
            k += 1
            if np.sum(np.abs(x - xold)) < tol or k >= max_iter:
                break
        # normalizing required in RB
        # x = x / np.sum(x) if normalize else x

    else:
        x = np.linalg.solve(Q, R)

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
            tol: int = 1e-7, max_iter: int = 200) -> np.ndarray:
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

    return x


def _dykstra(v: np.ndarray,
             func_list: list | tuple,
             tol=1e-7, max_iter=200,
             verbose=False) -> np.ndarray:
    k = 0
    v_list = np.zeros(shape=[len(func_list) + 1, len(v)])  # +1 is for v_0 and z_0 (but z_0 is redundant, just for consistency)
    z_list = np.zeros(shape=[len(func_list) + 1, len(v)])  # z_list[0] not used.
    v_list[-1] = v
    while True:
        vold_list = v_list.copy()
        zold_list = z_list.copy()
        v_list[0] = vold_list[-1]  # v_0 <- vold_m
        for i in range(len(v_list) - 1):  # i: 0 ~ m-1
            v_list[i + 1] = func_list[i](v_list[i] + zold_list[i + 1])
            z_list[i + 1] = v_list[i] + zold_list[i + 1] - v_list[i + 1]

        if np.sum((v_list[-1] - vold_list[-1]) ** 2) <= tol or k >= max_iter:
            if verbose:
                print('\tDykstra: Iter exceeds max_iter. The solution may be incorrect.') if k >= max_iter else print(f'\tDykstra: Converged after {k} iterations.')
            break

        k += 1

    return v_list[-1]


def _prox_l1_penalty(v: np.ndarray,
                     sigma: float, g: np.ndarray,
                     phi: float = 1.0) -> np.ndarray:
    return pyprox.L1(sigma=sigma, g=g).prox(v, phi)


def _prox_simplex(v: np.ndarray,
                  budget: float = 1.0) -> np.ndarray:
    return pyprox.Simplex(len(v), radius=budget, engine='numba').prox(v, 1.0)


def _prox_box(v: np.ndarray,
              lower: float | np.ndarray = -np.inf, upper: float | np.ndarray = np.inf,
              phi: float = 1.0) -> np.ndarray:
    return pyprox.Box(lower=lower, upper=upper).prox(v, phi)


@njit
def _prox_hyperplane(v: np.ndarray,
                     a: np.ndarray, b: float) -> np.ndarray:
    return v - (np.dot(a, v) - b) / np.linalg.norm(a, 2)**2 * a


@njit
def _prox_halfspace(v: np.ndarray,
                    c: np.ndarray, d: float) -> np.ndarray:
    return v - (np.maximum(np.dot(c, v) - d, 0) / np.linalg.norm(c, 2)**2) * c


def _prox_l1_ball(v: np.ndarray,
                  radius: float,
                  phi: float = 1.0) -> np.ndarray:
    return pyprox.L1Ball(len(v), radius).prox(v, phi)


def _prox_l2_ball(v: np.ndarray,
                  radius: float = 1.0,
                  phi: float = 1.0) -> np.ndarray:
    return v - pyprox.L2(sigma=2*radius).prox(v, phi)


def prox_fy(y: np.ndarray,
            cov: np.ndarray | None = None,
            current_pf: np.ndarray | None = None,
            benchmark_pf: np.ndarray | None = None,
            reference_pf: np.ndarray | None = None,
            penalty_current_l1: float = 0.0,
            penalty_refer_l1: float = 0.0,
            const: dict | None = None,
            phi: float = 1.0,
            tol: float = 1e-7, max_iter: int = 200,
            verbose_dykstra=False) -> np.ndarray:
    cov = np.identity(len(y)) if cov is None else cov
    benchmark_pf = np.zeros_like(y) if benchmark_pf is None else benchmark_pf
    current_pf = np.zeros_like(y) if current_pf is None else current_pf
    reference_pf = np.zeros_like(y) if reference_pf is None else reference_pf
    const_default = {
        'Budget': 1.0,
        'Weight bounds': {
            'Lower': -np.inf,
            'Upper': np.inf,
        },
        'Industry limit': {
            'Industry': [],
            'Lower': [],
            'Upper': [],
        },
        'Group limit': {
            'Group': [],
            'Lower': [],
            'Upper': [],
        },
        'Sector limit': {
            'Sector': [],
            'Lower': [],
            'Upper': [],
        },
        'Country limit': {
            'Country': [],
            'Lower': [],
            'Upper': [],
        },
        'Turnover limit': 1.0,
        # 'Transaction costs limit': None,
        'Leverage limit': 1.0,
        'Long/short exposure': {
            'Long': 1.0,
            'Short': -1.0
        },
        'Benchmarking': 1.0,
        'Tracking error floor': 0.0,
        'Active share floor': 0.0,
    #     'Num of active bets': None
    }

    if const is not None:
        for key in const_default.keys():
            const[key] = const_default[key] if key not in const.keys() else const[key]
    else:
        const = const_default

    _prox_dict = {
        'L1 Penalty': [lambda v: _prox_l1_penalty(v, penalty_current_l1, current_pf, phi),
                       lambda v: _prox_l1_penalty(v, penalty_refer_l1, reference_pf, phi)],
        'Budget': [lambda v: _prox_hyperplane(v, np.ones_like(v), const['Budget'])],
        'Weight bounds': [lambda v: _prox_box(v, const['Weight bounds']['Lower'], const['Weight bounds']['Upper'])],
        'Industry limit': [],
        'Group limit': [],
        'Sector limit': [],
        # 임시용
        'Country limit': [lambda v: _prox_halfspace(v, -1*np.array([0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.]), -1*const['Country limit']['Lower'][0]),
                          lambda v: _prox_halfspace(v, np.array([0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.]), const['Country limit']['Upper'][0])],
        'Turnover limit': [lambda v: _prox_l1_ball(v - current_pf, const['Turnover limit'])],
        'Leverage limit': [lambda v: _prox_l1_ball(v, const['Leverage limit'])],
        'Long/short exposure': [lambda v: _prox_halfspace(v, np.ones_like(v), const['Long/short exposure']['Long']),
                                lambda v: _prox_halfspace(v, np.ones_like(v), const['Long/short exposure']['Short'])],
        'Benchmarking': [lambda v: _prox_l2_ball(np.linalg.cholesky(cov).T@(v-benchmark_pf), const['Benchmarking'])],
        'Tracking error floor': [lambda v: _prox_l2_ball(-1*np.linalg.cholesky(cov).T@(v-benchmark_pf), -1*const['Tracking error floor'])],
        'Active share floor': [lambda v: _prox_l1_ball(-0.5*(v - benchmark_pf), -1*const['Active share floor'])],
    }
    for key in const:
        if const[key] is None:
            _prox_dict[key] = []
    _prox = sum(_prox_dict.values(), start=[])
    y = _dykstra(y, _prox, tol, max_iter, verbose_dykstra)
    return y


def _admm(er: np.ndarray, cov: np.ndarray,
          model: str, params_model: dict,
          constraints: dict,
          tol: float = 1e-7, max_iter: int = 1e+3,
          verbose=False,
          verbose_dykstra=False) -> np.ndarray:
    k = 0
    x = np.zeros_like(er)
    y = np.zeros_like(er)
    u = np.zeros_like(er)
    phi = 1.0
    while True:
        xold = x.copy()
        yold = y.copy()

        # ADMM
        x = prox_fx(x=y - u, er=er, cov=cov, phi=phi, **params_model[model]['fx'])
        y = prox_fy(x + u, phi=phi, **params_model[model]['fy'], const=constraints,
                    verbose_dykstra=verbose_dykstra)
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

        if verbose:
            verbose_cycle = 5
            if k % verbose_cycle == 0:
                print(f'° iter {k}')
                print('\tweights\t\t\t\t', np.round(x, 4) * 100)
                print('\trisk contributions\t', np.round((x * (cov @ x)) / (x.T @ cov @ x), 4) * 100)

        if max(sum((xold - x) ** 2),
               sum((yold - y) ** 2),
               sum((x - y) ** 2)) <= tol or k >= max_iter:
            if verbose:
                print('=' * 100)
                print('ADMM: Iter exceeds max_iter. The solution may be incorrect.') if k >= max_iter else print(
                    f'ADMM: Converged after {k} iterations.')
            break
        k += 1

    return x


def solve(er: np.ndarray, cov: np.ndarray,
          model: str, params_model: dict,
          constraints: dict,
          tol: float = 1e-7, max_iter: int = 1e+3) -> np.ndarray:
    def _bisect_constraint(lamb):
        params_model[model]['fx']['param_rb'] = lamb
        x = _admm(er=er, cov=cov,
                  model=model, params_model=params_model,
                  constraints=constraints,
                  tol=tol, max_iter=max_iter)
        return np.sum(x) - 1

    if params_model[model]['fx']['param_rb'] > 0.0:
        constraints['Budget'] = None    # budget constraint must be turned off.
        lambda_star = bisect(f=_bisect_constraint, a=0, b=20)
        params_model[model]['fx']['param_rb'] = lambda_star

    x = _admm(er=er, cov=cov,
              model=model, params_model=params_model,
              constraints=constraints,
              verbose=True,
              verbose_dykstra=True)

    return x