import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from traceback import print_exc
from multifidelity_studies.models.run_functions import CCBlade, OpenFAST


n_dims = 2

CC = CCBlade('cc_results_nd.pkl')
OF = OpenFAST('of_results_nd.pkl')

lofi_function = CC.run_vec
hifi_function = OF.run_vec

# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(11)

x_init = np.random.rand(3, n_dims) + 0.5
x = x_init.copy()
print()
print(x_init)

for i in range(21):
    y_low = lofi_function(x)
    y_high = hifi_function(x)

    # Construct RBF interpolater for error function
    differences = y_high - y_low
    
    input_arrays = np.split(x, x.shape[1], axis=1)
    
    input_arrays = [x.flatten() for x in input_arrays]
    
    try:
        e = Rbf(*input_arrays, differences, epsilon=0.1)
    except:
        print_exc()
        print("Done!")
        break

    # Create m_k = lofi + RBF
    def m(x):
        print('desvar: ', x)
        return -(lofi_function(np.atleast_2d(x)) + e(*x))
        
    n_plot = 9
    x_plot = np.linspace(0.5, 1.5, n_plot)
    X, Y = np.meshgrid(x_plot, x_plot)
    
    x_values = np.vstack((X.flatten(), Y.flatten())).T
    y_plot_high = hifi_function(x_values).reshape(n_plot, n_plot)

    plt.figure()
    plt.contourf(X, Y, y_plot_high, label='high')
    plt.scatter(x[:, 0], x[:, 1], color='k')
    
    plt.savefig(f'image_{i}.png')

    x0 = x[-1, :]
    res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=[(0.5, 1.5), (0.5, 1.5)])
    x_new = np.atleast_2d(res.x)
    
    x = np.vstack((x, x_new))
    
