import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from multifidelity_studies.models.testbed_components import simple_1D_low, simple_1D_high


func_low = simple_1D_low
func_high = simple_1D_high

# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(3)

# Misc utilities
def plot_functions():
    x_plot = np.linspace(0., 1., 101)
    surrogate = m(x_plot)
    y_plot_low = func_low(x_plot)
    y_plot_high = func_high(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot_low, label='low')
    plt.plot(x_plot, y_plot_high, label='high')
    plt.plot(x_plot, surrogate, label='mixed surrogate')
    plt.scatter(x, y_high, color='k')
    plt.legend()
    plt.show()

max_trust_radius = 1000.0
eta = 0.15
gtol = 1e-4
trust_radius = 0.1

# 1. Compute a step, s_k, that satisfies the fraction of Cauchy decrease requirement by solving:

# Query hifi and lofi in same places
x_all = np.random.rand(3)
x = x_all
y_low = func_low(x)
y_high = func_high(x)

# Construct RBF interpolater for error function
differences = y_high - y_low
e = Rbf(x, differences)

# Create m_k = lofi + RBF
m = lambda x: simple_1D_low(x) + e(x)

# plot_functions()

for i in range(10):
    # min (m_k(x_k + s_k)) st ||x_k|| <= del K
    x0 = x[-1]
    bounds = [(max(x0 - trust_radius, 0.), min(1., x0 + trust_radius))]
    res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=bounds)
    x_new = res.x
    x = np.hstack((x, x_new))
    y_low = func_low(x)
    
    # 2. If f_high(x + x_k) hasn't been evaluated, do it and save it
    y_high = func_high(x)
    
    # 3. Compute the ratio of actual improvement to predicted improvement
    actual_reduction = func_high(x[-2]) - func_high(x_new)
    predicted_reduction = func_high(x[-2]) - m(x_new)
    
    # 4. Accept or reject the trial point according to that ratio
    if predicted_reduction <= 0:
        print('not enough reduction! probably at a local optimum')
        break
        
    rho = actual_reduction / predicted_reduction

    hits_boundary = False
    # 5. Update trust region according to rho_k
    if rho < 0.25:
        trust_radius *= 0.25
    elif rho > 0.75:  # and hits_boundary:
        trust_radius = min(2*trust_radius, max_trust_radius)

    # 6. Create a new model m_k+1 using Algo 2.2
    differences = y_high - y_low
    e = Rbf(x, differences)

    # Create m_k = lofi + RBF
    m = lambda x: simple_1D_low(x) + e(x)

    print(x)

    plot_functions()
