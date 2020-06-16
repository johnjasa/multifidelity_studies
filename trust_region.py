import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from testbed_components import simple_1D_low, simple_1D_high


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(3)

# Misc utilities
def plot_functions():
    x_plot = np.linspace(0., 1., 101)
    surrogate = m(x_plot)
    y_plot_low = simple_1D_low(x_plot)
    y_plot_high = simple_1D_high(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot_low, label='low')
    plt.plot(x_plot, y_plot_high, label='high')
    plt.plot(x_plot, surrogate, label='mixed surrogate')
    plt.scatter(x, y_high, color='k')
    plt.legend()
    plt.show()

# 1. Compute a step, s_k, that satisfies the fraction of Cauchy decrease requirement by solving:
# min (m_k(x_k + s_k)) st ||x_k|| <= del K

# Construct m_k

# Query hifi and lofi in same places
x_all = np.random.rand(20)
for i in range(2, len(x_all)):
    x = x_all[:i]
    y_low = simple_1D_low(x)
    y_high = simple_1D_high(x)

    # Construct RBF interpolater for error function
    differences = y_high - y_low
    e = Rbf(x, differences)

    # Create m_k = lofi + RBF
    m = lambda x: simple_1D_low(x) + e(x)

    plot_functions()

# 2. If f_high(x + x_k) hasn't been evaluated, do it and save it

# 3. Compute the ratio of actual improvement to predicted improvement

# 4. Accept or reject the trial point according to that ratio

# 5. Update trust region according to rho_k

# 6. Create a new model m_k+1 using Algo 2.2

# 7. Check for convergence



