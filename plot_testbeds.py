import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from testbed_components import simple_one_D


x_low = np.linspace(0., 1., 11)
x_high = np.array([0., 0.4, 0.6, 1.0])

prob = om.Problem()
prob.model.add_subsystem('low', simple_one_D(fidelity='low', n_pts=len(x_low)))
prob.model.add_subsystem('high', simple_one_D(fidelity='high', n_pts=len(x_high)))

prob.setup()

prob['low.x'] = x_low
prob['high.x'] = x_high

prob.run_model()

y_low = prob['low.y']
y_high = prob['high.y']


x_full = np.linspace(0., 1., 101)
prob = om.Problem()
prob.model.add_subsystem('low', simple_one_D(fidelity='low', n_pts=len(x_full)))
prob.model.add_subsystem('high', simple_one_D(fidelity='high', n_pts=len(x_full)))

prob.setup()

prob['low.x'] = x_full
prob['high.x'] = x_full

prob.run_model()

y_full_low = prob['low.y']
y_full_high = prob['high.y']


plt.figure()
plt.plot(x_full, y_full_low, label='low-fidelity')
plt.scatter(x_low, y_low)

plt.plot(x_full, y_full_high, label='high-fidelity')
plt.scatter(x_high, y_high)


plt.show()
