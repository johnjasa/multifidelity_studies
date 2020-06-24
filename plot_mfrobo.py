import numpy as np
import dill
import matplotlib.pyplot as plt
from testbed_components import simple_1D_low, simple_1D_medium, simple_1D_high


colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

funcs = [simple_1D_high, simple_1D_medium, simple_1D_low]
labels = ['High fidelity', 'Medium fidelity', 'Low fidelity']

filename = "mfrobo_out.pkl"

with open(filename, "rb") as f:
    mfrobo = dill.load(f)

plt.figure(figsize=(6, 4))
x = np.linspace(0.0, 1.0, 201)
for j, func in enumerate(funcs):
    plt.plot(x, func(x) + func(0) / 2.0, color=colors[j], label=labels[j])

for i, func_vals in enumerate(mfrobo.fB_all):
    design_vector = mfrobo.D_all[i]
    for j, num_calls in enumerate(mfrobo.m_star_all[i]):
        obj_vals = func_vals[:num_calls, j]
        if num_calls > 0:
            plt.scatter(
                np.ones(num_calls) * design_vector, obj_vals, color=colors[j], alpha=0.3, linewidths=0.,
            )

plt.xlim([0., 1.])
plt.xlabel('Design variable, x')
plt.ylabel('Function output, y')

plt.legend()

plt.tight_layout()
plt.savefig('simple_mfrobo.png', dpi=600)
# plt.show()
