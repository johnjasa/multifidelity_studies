import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from testbed_components import simple_1D, simple_1D_low, simple_1D_high


x_low = np.linspace(0., 1., 11)
x_high = np.array([0., 0.4, 0.6, 1.0])

y_low = simple_1D_low(x_low)
y_high = simple_1D_high(x_high)

x_full = np.linspace(0., 1., 101)
y_full_low = simple_1D_low(x_full)
y_full_high = simple_1D_high(x_full)


plt.figure()
plt.plot(x_full, y_full_low, label='low-fidelity')
plt.scatter(x_low, y_low)

plt.plot(x_full, y_full_high, label='high-fidelity')
plt.scatter(x_high, y_high)


plt.show()
