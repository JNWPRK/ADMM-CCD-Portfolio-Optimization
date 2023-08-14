# Unified Portfolio Optimization via ADMM-CCD

# Problem Setting (Robo-advisor optimization problem)

## Objective function

![image](https://github.com/NeurofusionAI/quant-intern/assets/87808232/6697de26-9b86-48a7-b830-1b61389b55b5)

## Constraints set ($\Omega$)

![image](https://github.com/NeurofusionAI/quant-intern/assets/87808232/52390a06-fd2c-4010-89a8-a19b1702e48e)

## Methodology

### Splitting functions for ADMM

![image](https://github.com/NeurofusionAI/quant-intern/assets/87808232/fc55bb16-299e-40bf-8216-5f908fcdb329)

First, split $f_{Robo}(x)$ into $f_{MVO}(x)$, $f_{\ell_1}(x)$, $f_{\ell_2}(x)$, $f_{RB}(x)$ and $\textbf{1}_{\Omega_0}(x)$, $\textbf{1}_{\Omega}(x)$.

$\textbf{1}_{\Omega}(x)$ is an indicator function, meaning that $\textbf{1}_{\Omega}(x)=0$ for $x \in \Omega$ and $\textbf{1}_{\Omega}(x) = +\infin$ for $x \notin \Omega$. 

Then, set $f_x(x)$ and $f_y(y)$ by arranging those functions above, and impose a constraint of $x=y$ which guarantees $x$ and $y$ converge to the same value.

![image](https://github.com/NeurofusionAI/quant-intern/assets/87808232/f2f0ae7d-4362-4468-a3fa-c1f30ebd31b0)

$f_x(x)$ and $f_y(y)$ are ingredients for the **ADMM algorithm.**

### ADMM algorithm

Before explaining the ADMM algorithm, the **proximal operator** is introduced first.

$$
\text{prox}_{f}(v)=x^{\star}=\text{argmin}_x \biggl[ f(x)+ {1\over2}\|x-v\|_2^2 \biggr]
$$

By using the proximal operators of $f_x(x)$ and $f_y(y)$, we can express each step on the ADMM algorithm.

$$
{x^{k+1}=\text{prox}_{\varphi f_x} (y^k - u^k)} \\ {y^{k+1}=\text{prox}_{\varphi f_y} (x^{k+1} + u^k)} \\ {u^{k+1} = u^k + x^{k+1} - y^{k+1}}
$$

Iterate these steps until $x$ and $y$ converge. 

$\varphi$ is a penalization parameter of ADMM, and it is internally adjusted (Perrin, Roncalli, 2019, p.56).

Now we need to calculate $\text{prox}$ of $f_x(x)$ and $f_y(y)$. We adopt CCD algorithm to the former, and the analytical form of proximal operators to the latter.

- ${x^{k+1}=\text{prox}_{\varphi f_x} (y^k - u^k)}$: CCD algorithm
- ${y^{k+1}=\text{prox}_{\varphi f_y} (x^{k+1} + u^k)}$: Analytical forms

### $x$-update: CCD algorithm

Perrin et al. (2019) proposed the $x$-update formula.

![image](https://github.com/NeurofusionAI/quant-intern/assets/87808232/6539ab0b-f669-4914-a13e-925dd000b005)

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

All these forms have the analytical form the of proximal operator. 

(or in the Python package, PyProximal [https://pyproximal.readthedocs.io/en/stable/index.html](https://pyproximal.readthedocs.io/en/stable/index.html))

### Dykstra’s algorithm

However, there are some proximal operators which do not have analytical form, such as inequality linear constraint $Cx \le D$ and an additive form of several functions $\text{prox}_{f_1 + f_2 + … + f_m}(v)$.

In this case, you can use Dykstra’s algorithm.

![image](https://github.com/NeurofusionAI/quant-intern/assets/87808232/f63514af-ddf0-4562-90fe-25eb7be96316)

To put it simply, splitting a function or a set to $m$ parts and iterating those with a dual variable, $z$.

 

### ** Points to note for optimizing risk budgeting portfolio

When you optimize RB portfolio, the budget constraint $\textbf{1}^T x = 1$ **must not be imposed.**

(while the long-only condition satisfies automatically thanks to the property of the log-barrier function)

You will find the log-barrier parameter $\lambda^{\star}$ with bisect optimization, which makes the sum of $x$ elements, in the middle of ADMM algorithm.

After finding it, you implement ADMM again using $\lambda^{\star}$.

![image](https://github.com/NeurofusionAI/quant-intern/assets/87808232/c09a8496-939a-43ab-8e91-73eb5564e8cb)

 

# Reference

- Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles (Richard, Roncalli, 2019)
- Machine Learning Optimization Algorithms & Portfolio Allocation (Perrin, Roncalli, 2019)