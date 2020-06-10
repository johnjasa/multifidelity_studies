from __future__ import print_function

# OAS problem MFROBO
import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict
from mfrobo_class import MFROBO
from testbed_components import simple_1D_low, simple_1D_medium, simple_1D_high


np.random.seed(314)

# Fidelity parameters
funcs = [simple_1D_high, simple_1D_medium, simple_1D_low]

Din = np.array([0.2])

Ex_stdx = OrderedDict()

Ex_stdx["dummy"] = (0.0, 0.1)

# Weight for variance in the objective function
eta = 0

# Target MSE for moment estimates
J_star = 1e-3

mfrobo_inst = MFROBO(funcs, Ex_stdx, eta, J_star, "out2.pkl")
mfrobo_inst.t_DinT = np.array([0.5, 0.1, 0.05])

res = minimize(
    mfrobo_inst.obj_func,
    Din,
    args=(),
    method="COBYLA",
    tol=1e-10,
    options={"disp": True, "maxiter": 1000, "rhobeg": 0.1},
)
