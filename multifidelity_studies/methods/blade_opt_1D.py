import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
from traceback import print_exc
from multifidelity_studies.models.run_functions import CCBlade, OpenFAST



# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(111)

num_initial_points = 3
x_init = np.random.rand(num_initial_points) + 0.5
x = x_init.copy()
print()
print(x_init)

list_of_desvars = []
for i in range(num_initial_points):
    desvars = OrderedDict()
    desvars['blade.opt_var.chord_opt_gain'] = x[i]
    list_of_desvars.append(desvars)

CC = CCBlade(desvars, 'cc_results.pkl')
OF = OpenFAST(desvars, 'of_results.pkl')

lofi_function = CC.run_vec
hifi_function = OF.run_vec

for i in range(21):
    y_low = lofi_function(list_of_desvars)
    y_high = hifi_function(list_of_desvars)
    
    # Construct RBF interpolater for error function
    differences = y_high - y_low
    
    try:
        e = Rbf(x, differences, epsilon=0.1)
    except:
        print_exc()
        print("Done!")
        break

    # Create m_k = lofi + RBF
    def m(x):
        desvars = CC.unflatten_desvars(x)
        return -(lofi_function([desvars]) + e(x))

    x_plot = np.linspace(0.5, 1.5, 21)
    
    list_of_desvars_plot = []
    surrogate = []
    for j in range(21):
        desvars = OrderedDict()
        desvars['blade.opt_var.chord_opt_gain'] = x_plot[j]
        list_of_desvars_plot.append(desvars)
        surrogate.append(-m(x_plot))
        
    surrogate = np.array(surrogate)
    y_plot_low = lofi_function(list_of_desvars_plot)
    y_plot_high = hifi_function(list_of_desvars_plot)

    plt.figure()
    plt.plot(x_plot, y_plot_low, label='low')
    plt.plot(x_plot, y_plot_high, label='high')
    plt.plot(x_plot, surrogate, label='mixed surrogate')
    plt.scatter(x, y_high, color='k')
    
    plt.xlim([0.5, 1.5])
    plt.ylim([0.4, 0.5])
    plt.legend()
    plt.savefig(f'image_1D_{i}.png')
    

    x0 = x[-1]
    res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=[(0.5, 1.5)])
    x_new = res.x
    x = np.hstack((x, x_new))
    
    desvars = OrderedDict()
    desvars['blade.opt_var.chord_opt_gain'] = x_new
    list_of_desvars.append(desvars)
    