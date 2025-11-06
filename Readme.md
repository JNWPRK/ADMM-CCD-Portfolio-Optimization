# Unified Portfolio Optimization via ADMM-CCD

# Problem Setting (Robo-advisor optimization problem)

## Objective function

$$ x_{t+1}^{\star} = \operatorname{argmin}_x f_{Robo}(x) $$
$$ \text{s.t.} \begin{cases} \textbf{1}_n x = 1 \\ \textbf0_n \le x \le \textbf{1}_n \\ x \in \Omega \end{cases} $$

where:
$$ f_{Robo}(x) = \gamma (x-b)^T \Sigma (x-b) - \eta(x-b)^T\mu + \varrho_1 \|x-x_t\|_1 + {1 \over 2} \varrho_2 \| x-x_t\|_2 ^2 + \tilde{\varrho}_1 \|x-\tilde{x}\|_1 + {1\over 2} \tilde{\varrho}_2 \|x-\tilde{x}\|_2 ^2 - \lambda \sum_{i=1} ^n \mathcal{RB_i}\text{ln} \cdot x_i $$

## Constraints set ($\Omega$)

| Constraints           | Formula                                                                         |
| --------------------- | ------------------------------------------------------------------------------- |
| No cash and leverage  | $\sum_{i=1} ^n x_i = 1$                                                         |
| No short selling      | $x_i \ge 0$                                                                     |
| Weight bounds         | $x_i ^- \le x_i \le x_i ^+$                                                     |
| Asset class limits    | $c_j ^- \le \sum_{i\in C_j}x_i \le c_j ^+$                                      |
| Turnover              | $\sum_{i=1} ^n \|x-\tilde{x}_i\| \le \tau ^+$                                   |
| Transaction costs     | $\sum_{i=1} ^n (c_i ^-(\tilde{x}_i - x)_+ + (c_i ^+(x-\tilde{x}_i)_+) \le c ^+$ |
| Leverage limit        | $\sum_{i=1} ^n \|x_i\|\le \mathcal{L}^+$                                        |
| Long/short exposure   | $-\mathcal{LS}^- \le \sum_{i=1} ^n x_i \le \mathcal{LS}^+$                      |
| Benchmarking          | $\sqrt{(x-\tilde{x})^T \Sigma (x-\tilde{x})} \le \sigma ^+$                     |
| Tracking error floor  | $\sqrt{(x-\tilde{x})^T \Sigma (x-\tilde{x})} \ge \sigma ^-$                     |
| Active share floor    | $\sum_{i=1} ^n$                                                                 |
| Number of active bets | $(x^T x)^{-1} \ge \mathcal{N} ^-$                                               |

## Methodology

### Splitting functions for ADMM

$$ f_{Robo}(x) = f_{MVO}(x) + f_{\ell_1}(x) + f_{\ell_2}(x) + f_{RB}(X)+\textbf1_{\Omega_0}(x) + \textbf1_{\Omega}(x) $$
where:
$$ f_{MVO}(x) = \gamma (x-b)^T \Sigma (x-b) - \eta(x-b)^T\mu $$
$$ f_{\ell_1}(x) =\varrho_1 \|x-x_t\|_1 + \tilde{\varrho}_1 \|x-\tilde{x}\|_1 $$
$$ f_{\ell_2}(x) = {1 \over 2} \varrho_2 \| x-x_t\|_2 ^2 + {1\over 2} \tilde{\varrho}_2 \|x-\tilde{x}\|_2 ^2 $$
$$ f_{RB}(x) = - \lambda \sum_{i=1} ^n \mathcal{RB_i}\text{ln} \cdot x_i $$
$$ \Omega_0 = \{x\in [0,1]^n : \textbf1_n^Tx=1\} $$

First, split $f_{Robo}(x)$ into $f_{MVO}(x)$, $f_{\ell_1}(x)$, $f_{\ell_2}(x)$, $f_{RB}(x)$ and $\textbf{1}_{\Omega_0(x)}$, $\textbf{1}_{\Omega(x)}$.

$\textbf{1}_{\Omega}(x)$ is an indicator function, meaning that $\textbf{1}_{\Omega}(x)=0$ for $x \in \Omega$ and $\textbf{1}_{\Omega}(x) = +\infty$ for $x \notin \Omega$. 

Then, set $f_x(x)$ and $f_y(y)$ by arranging those functions above, and impose a constraint of $x=y$ which guarantees $x$ and $y$ converge to the same value.

$$ f_{x}(x) = f_{MVO}(x) + f_{\ell_2}(x) + f_{RB}(X) $$
$$ f_{y}(y) = f_{\ell_1}(y) + \textbf1_{\Omega_0}(y) + \textbf1_{\Omega}(y) $$

$f_x(x)$ and $f_y(y)$ are ingredients for the **ADMM algorithm.**
$$ \{x^\star, y^\star \} = \operatorname{argmin}_{x,y} f_x(x)+f_y(y) $$
$$ \text{s.t.} \quad x=y $$

### ADMM algorithm

Before explaining the ADMM algorithm, the **proximal operator** is introduced first.

$$
\text{prox}_{f}(v)=x^{\star}=\text{argmin}_x \biggl[ f(x)+ {1\over2}\|x-v\|_2^2 \biggr]
$$

By using the proximal operators of $f_x(x)$ and $f_y(y)$, we can express each step on the ADMM algorithm.

$$
{x^{k+1}=\text{prox}\_{\varphi f_x} (y^k - u^k)} \\ {y^{k+1}=\text{prox}_{\varphi f_y} (x^{k+1} + u^k)} \\ {u^{k+1} = u^k + x^{k+1} - y^{k+1}}
$$

Iterate these steps until $x$ and $y$ converge. 

$\varphi$ is a penalization parameter of ADMM, and it is internally adjusted (Perrin, Roncalli, 2019, p.56).

Now we need to calculate $\text{prox}$ of $f_x(x)$ and $f_y(y)$. We adopt CCD algorithm to the former, and the analytical form of proximal operators to the latter.

- ${x^{k+1}=\text{prox}_{\varphi f_x} (y^k - u^k)}$: CCD algorithm
- ${y^{k+1}=\text{prox}_{\varphi f_y} (x^{k+1} + u^k)}$: Analytical forms

### $x$-update: CCD algorithm

Perrin et al. (2019) proposed the $x$-update formula.
$$x_i^{(k+1)} = \frac{R_i - \sum_{j<i} x_j^{(k+1)} Q_{i,j} - \sum_{j>i} x_j^{(k)} Q_{i,j}}{2 Q_{i,i}} + \frac{\sqrt{\left(\sum_{j<i} x_j^{(k+1)} Q_{i,j} + \sum_{j>i} x_j^{(k)} Q_{i,j} - R_i\right)^2 + 4 \lambda_i Q_{i,i}}}{2 Q_{i,i}}$$
$$
$$
where the matrices $Q$ and $R$ are defined as:
$$
$$
$$Q = \mathbf{\Sigma}_t + \varrho_2 \mathbf{\Gamma}_2^\top \mathbf{\Gamma}_2 + \tilde{\varrho}_2 \tilde{\mathbf{\Gamma}}_2^\top \tilde{\mathbf{\Gamma}}_2 + \varphi \mathbf{I}_n$$

$$R = \gamma \boldsymbol{\mu}_t + \mathbf{\Sigma}_t \mathbf{b} + \varrho_2 \mathbf{\Gamma}_2^\top \mathbf{\Gamma}_2 \mathbf{x}_t + \tilde{\varrho}_2 \tilde{\mathbf{\Gamma}}_2^\top \tilde{\mathbf{\Gamma}}_2 \tilde{\mathbf{x}} + \varphi \left(\mathbf{y}^{(k)} - \mathbf{u}^{(k)}\right)$$

It is derived from the Cyclical-Coordinate Descent algorithm. 
(You can get this by simply differentiating $f_x(x)$ with respect to $x$.)

### $y$-update: Analytical forms of various proximal operators

We can express all the constraints belonging to the set $\Omega$ to one of these forms below.

- Simplex ($\textbf{1}^T x = 1$ and $x \ge 0$)
- Box ($x^{-} \le x \le x^{+}$)
- Affine set ($Ax=B$)
- Halfspace ($c^T x = d$)
- L1 ($\|x\|_1$)
- L1 ball ($\|x\|_1 \le r$)
- L2 ($\|x\|_2 ^2$)
- L2 ball ($\|x\|_2 ^2 \le r$)

All these forms have the analytical form of the proximal operator. 

(or in the Python package, PyProximal [https://pyproximal.readthedocs.io/en/stable/index.html](https://pyproximal.readthedocs.io/en/stable/index.html))

### Dykstra’s algorithm

However, there are some proximal operators which do not have analytical form, such as inequality linear constraint $Cx \le D$ and an additive form of several functions $\text{prox}_{f_1 + f_2 + … + f_m}(v)$.

In this case, you can use Dykstra’s algorithm.

The Dykstra's algorithm is particularly efficient when we consider the projection problem:
$$ x^{\star} = \mathcal{P}_{\Omega}(\mathbf{v})$$ 
where: 
$$\Omega = \Omega_1 \cap \Omega_2 \cap \dots \cap \Omega_m$$
Indeed, the Dykstra's algorithm becomes:
- The ${x}$-update is:
$${x}^{(k+1, j)} = \operatorname{prox}_{f_j}\left({x}^{(k+1, j-1)} + {z}^{(k, j)}\right) = \mathcal{P}_{\Omega_j}\left({x}^{(k+1, j-1)} + {z}^{(k, j)}\right) $$
- The ${z}$-update is:
$${z}^{(k+1, j)} = {x}^{(k+1, j-1)} + {z}^{(k, j)} - {x}^{(k+1, j)}$$
where ${x}^{(1, 0)} = {v}$, ${z}^{(k, j)} = \mathbf{0}_n$ for $k = 0$ and ${x}^{(k+1, 0)} = {x}^{(k, m)}$


To put it simply, splitting a function or a set to $m$ parts and iterating those with a dual variable, $z$. 

### Points to note for optimizing risk budgeting portfolio

When you optimize RB portfolio, the budget constraint $\textbf{1}^T x = 1$ **must not be imposed.**

(while the long-only condition satisfies automatically thanks to the property of the log-barrier function)

You will find the log-barrier parameter $\lambda^{\star}$ with bisect optimization, which makes the sum of $x$ elements to $1$, in the middle of ADMM algorithm.

After finding it, you implement ADMM again using $\lambda^{\star}$.

>**Algorithm**: General algorith for computing the constrained RB portfolio
>
>The goal is to compute the optimal Lagrange multiplier $\lambda^{\star}$ and the solution $x^{\star}(\mathcal{S}, \Omega)$
>We consider two scalars $a_{\lambda}$ and $b_{\lambda}$ such that $a_{\lambda} < b_{\lambda}$ and $\lambda^{\star} \in [a_{\lambda}, b_{\lambda}]$
>We note $\varepsilon_{\lambda}$ the convergence criterion of the bisection algorith (e.g. $10^{-8}$)
>
>**repeat**
 >   We calculate $\lambda = \frac{a_\lambda + b_\lambda}{2}$
 >  We compute ${x}^{\star}(\lambda)$ the solution of the minimization problem:
 >       ${x}^{\star}(\lambda) = {\operatorname{argmin}} \mathcal{L}({x}; \lambda)$
 >   if $\sum_{i=1}^n x^{\star}_i(\lambda) < 1$ then
 >       $a_\lambda \leftarrow \lambda$
 >   else
 >       $b_\lambda \leftarrow \lambda$
 >   end if
>**until** $|\sum_{i=1}^n x^{\star}_i(\lambda) - 1| \leq \varepsilon_\lambda$
>return $\lambda^\star \leftarrow \lambda$ and ${x}^\star(\mathcal{S}, \Omega) \leftarrow {x}^\star(\lambda^\star)$

# Reference

- Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles (Richard, Roncalli, 2019)
- Machine Learning Optimization Algorithms & Portfolio Allocation (Perrin, Roncalli, 2019)
