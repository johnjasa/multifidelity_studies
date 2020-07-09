import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from traceback import print_exc
from multifidelity_studies.models.testbed_components import simple_2D_low, simple_2D_high


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(3)
func_low = simple_2D_low
func_high = simple_2D_high

x_init = np.random.rand(2, 3)
x = x_init.copy()

for i in range(20):
    y_low = func_low(x)
    y_high = func_high(x)

    # Construct RBF interpolater for error function
    differences = y_high - y_low
    
    input_arrays = np.split(x, x.shape[0], axis=0)
    
    input_arrays = [x.flatten() for x in input_arrays]
    
    try:
        e = Rbf(*input_arrays, differences, epsilon=0.1)
    except:
        print_exc()
        print("Done!")
        break

    # Create m_k = lofi + RBF
    def m(x):
        input_arrays = np.split(x, x.shape[0], axis=0)
        input_arrays = [x.flatten() for x in input_arrays]
        return func_low(x) + e(*input_arrays)

    n_plot = 101
    x_plot = np.linspace(0., 1., n_plot)
    X, Y = np.meshgrid(x_plot, x_plot)
    
    x_values = np.vstack((X.flatten(), Y.flatten()))
    # surrogate = m(x_values).reshape(n_plot, n_plot)
    # y_plot_low = func_low(x_values).reshape(n_plot, n_plot)
    y_plot_high = func_high(x_values).reshape(n_plot, n_plot)
    
    plt.figure()
    # plt.contourf(X, Y, surrogate, label='mixed surrogate')
    # plt.contourf(X, Y, y_plot_low, label='low')
    plt.contourf(X, Y, y_plot_high, label='high')
    plt.scatter(x[0], x[1], color='k')
    plt.show()
    

    x0 = x[:, -1]
    res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=[(0., 1.), (0., 1.)])
    x_new = np.atleast_2d(res.x).T
    x = np.hstack((x, x_new))
    
print()
print(f'Number of high-fidelity calls for MFM: {x.shape[1]}')
print(f'Answer found: {np.squeeze(x_new)}')
print()

res = minimize(func_high, x_init[:, 2], method='SLSQP', tol=1e-6, bounds=[(0., 1.), (0., 1.)])

print(f'Number of high-fidelity calls for hifi only: {res.nfev}, jac calls: {res.njev}')
print(f'Answer found: {res.x}')
print()
