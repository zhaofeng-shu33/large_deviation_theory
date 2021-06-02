import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# numerical example to show the trade-off between the two error exponents E_0 and E_1
p = 0.3
q = 0.8
eps = 1e-3
def psi_p(x):
    return np.power(p, 1 - x) * np.power(q, x) + \
    np.power(1 - p, 1 - x) * np.power(1 - q, x)
res = minimize(psi_p, [0.5])

_lambda = np.linspace(eps, 1-eps)
psi_p_lambda = np.log(np.power(p, 1 - _lambda) * np.power(q, _lambda) + \
    np.power(1 - p, 1 - _lambda) * np.power(1 - q, _lambda))
gamma = np.log(q / p) * np.power(p, 1 - _lambda) * np.power(q, _lambda) + \
    np.log( (1 - q) / (1 - p) ) * np.power(1 - p, 1 - _lambda) * np.power(1 - q, _lambda)
gamma /= np.exp(psi_p_lambda)
E_0 = _lambda * gamma - psi_p_lambda
E_1 = E_0 - gamma
max_E_1 = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
max_E_0 = q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
plt.plot(E_0, E_1)
plt.plot([0, 0.3], [0, 0.3], '--')
plt.text(0, max_E_1, 'D(P||Q)')
plt.text(max_E_0, 0, 'D(Q||P)')
res_y = -np.log(psi_p(res['x']))
plt.text(res_y, res_y, 'Chernoff Information')
plt.scatter([0, max_E_0, res_y], [max_E_1, 0, res_y], marker='o', c='red')
plt.xlabel('$E_0$')
plt.ylabel('$E_1$')
plt.title('Error exponents in hypothesis testing of two Bernoulli random variables')
plt.savefig('trade_off.eps')
plt.show()
