import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from collections import OrderedDict
import smt.surrogate_models as smt


class BaseMethod:
    """
    The base class that all multifidelity optimization methods inherit from.

    Attributes
    ----------
    bounds
    disp
    model_low
    model_high
    counter
    objective
    constraints
    n_dims
    x
    objective
    objective_scaler
    approximation_functions
    
    """

    def __init__(self, model_low, model_high, bounds, disp=True, num_initial_points=5):
        """
        Initialize the method by storing the models and populating the
        first points.
        
        Parameters
        ----------
        model_low : BaseModel instance
            The low-fidelity model instance provided by the user.
        model_high : BaseModel instance
            The high-fidelity model instance provided by the user.
        bounds : array
            A 2D array of design variable bounds, e.g. [[0., 1.], ...] for
            each design variable.
        disp : bool, optional
            If True, the method will print out progress and results to the terminal.
        num_initial_points : int
            The number of initial points to use to populate the surrogate-based
            approximations of the methods. In general, higher dimensional problems
            require more initial points to get a reasonable surrogate approximation.
        """

        self.bounds = np.array(bounds)
        self.disp = disp

        self.model_low = model_low
        self.model_high = model_high

        self.initialize_points(num_initial_points)
        self.counter = 0

        self.objective = None
        self.constraints = []

    def initialize_points(self, num_initial_points):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        self.n_dims = self.model_high.total_size
        x_init_raw = np.random.rand(num_initial_points, self.n_dims)
        self.x = (
            x_init_raw * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        )

    def set_initial_point(self, x):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        if isinstance(x, (float, list)):
            x = np.array(x)
        self.x = np.vstack((self.x, x))

    def add_objective(self, objective_name, scaler=1.0):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        self.objective = objective_name
        self.objective_scaler = scaler

    def add_constraint(self, constraint_name, equals=None, lower=None, upper=None):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        self.constraints.append(
            {"name": constraint_name, "equals": equals, "lower": lower, "upper": upper,}
        )

    def construct_approximations(self, interp_method="smt"):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        outputs_low = self.model_low.run_vec(self.x)
        outputs_high = self.model_high.run_vec(self.x)

        approximation_functions = {}
        outputs_to_approximate = [self.objective]

        if len(self.constraints) > 0:
            for constraint in self.constraints:
                outputs_to_approximate.append(constraint["name"])

        for output_name in outputs_to_approximate:
            differences = outputs_high[output_name] - outputs_low[output_name]

            if interp_method == "rbf":
                input_arrays = np.split(self.x, self.x.shape[1], axis=1)
                input_arrays = [x.flatten() for x in input_arrays]

                # Construct RBF interpolater for error function
                e = Rbf(*input_arrays, differences)

                # Create m_k = lofi + RBF
                def approximation_function(x):
                    return self.model_low.run(x)[output_name] + e(*x)

            elif interp_method == "smt":
                # sm = smt.RMTB(
                #     xlimits=self.bounds,
                #     order=3,
                #     num_ctrl_pts=5,
                #     print_global=False,
                # )
                sm = smt.RBF(print_global=False,)

                sm.set_training_values(self.x, differences)
                sm.train()

                def approximation_function(x, output_name=output_name, sm=sm):
                    return self.model_low.run(x)[output_name] + sm.predict_values(
                        np.atleast_2d(x)
                    )

            # Create m_k = lofi + RBF
            approximation_functions[output_name] = approximation_function

        self.approximation_functions = approximation_functions
