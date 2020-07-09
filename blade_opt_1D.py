import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from models.run_functions import CCBlade, OpenFAST


CC = CCBlade('cc_results.pkl')
OF = OpenFAST('of_results.pkl')

lofi_function = CC.run_vec
hifi_function = OF.run_vec

# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(111)


x_init = np.random.rand(3) + 0.5
x = x_init.copy()
print()
print(x_init)

for i in range(21):
    y_low = lofi_function(x)
    print(y_low)
    y_high = hifi_function(x)
    print(y_high)
    print()

    # Construct RBF interpolater for error function
    differences = y_high - y_low
    print(x)
    print(differences)
    print()
    
    try:
        e = Rbf(x, differences, epsilon=0.1)
    except:
        print("Done!")
        break

    # Create m_k = lofi + RBF
    m = lambda x: -(lofi_function(x) + e(x))

    x_plot = np.linspace(0.5, 1.5, 21)
    surrogate = -m(x_plot)
    y_plot_low = lofi_function(x_plot)
    y_plot_high = hifi_function(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot_low, label='low')
    plt.plot(x_plot, y_plot_high, label='high')
    plt.plot(x_plot, surrogate, label='mixed surrogate')
    plt.scatter(x, y_high, color='k')
    
    plt.xlim([0.5, 1.5])
    plt.ylim([0.4, 0.5])
    plt.legend()
    plt.savefig(f'image_{i}.png')
    

    x0 = x[-1]
    res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=[(0.5, 1.5)])
    x_new = res.x
    x = np.hstack((x, x_new))
    
