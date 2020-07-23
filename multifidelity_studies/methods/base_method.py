import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from collections import OrderedDict
import smt.surrogate_models as smt


class BaseMethod():
    """
    The base class that all multifidelity optimization methods inherit from.
    
    """
    def __init__(self, model_low, model_high, bounds, num_initial_points=5):
        
        self.bounds = np.array(bounds)
        
        self.model_low = model_low
        self.model_high = model_high
        
        self.initialize_points(num_initial_points)
        self.counter = 0
        
        self.objective = None
        self.constraints = []
        
    def initialize_points(self, num_initial_points):
        self.n_dims = self.model_high.total_size
        x_init_raw = np.random.rand(num_initial_points, self.n_dims)
        self.x = x_init_raw * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        
    def add_objective(self, objective_name, scaler=1.0):
        self.objective = objective_name
        self.objective_scaler = scaler
        
    def add_constraint(self, constraint_name, equals=None, lower=None, upper=None):
        self.constraints.append({'name' : constraint_name,
                                 'equals' : equals,
                                 'lower' : lower,
                                 'upper' : upper,
                                 })
                                 
    def construct_approximations(self, interp_method='smt'):
        outputs_low = self.model_low.run_vec(self.x)
        outputs_high = self.model_high.run_vec(self.x)
        
        approximation_functions = {}
        outputs_to_approximate = [self.objective]
        
        if len(self.constraints) > 0:
            for constraint in self.constraints:
                outputs_to_approximate.append(constraint['name'])
        
        for output_name in outputs_to_approximate:
            differences = outputs_high[output_name] - outputs_low[output_name]
            
            if interp_method == 'rbf':
                input_arrays = np.split(self.x, self.x.shape[1], axis=1)
                input_arrays = [x.flatten() for x in input_arrays]
                
                # Construct RBF interpolater for error function
                e = Rbf(*input_arrays, differences)
                
                # Create m_k = lofi + RBF
                def approximation_function(x):
                    return self.model_low.run(x)[output_name] + e(*x)
                    
            elif interp_method == 'smt':
                # sm = smt.RMTB(
                #     xlimits=self.bounds,
                #     order=3,
                #     num_ctrl_pts=5,
                #     print_global=False,
                # )
                sm = smt.RBF(
                    print_global=False,
                )
                
                sm.set_training_values(self.x, differences)
                sm.train()
                
                def approximation_function(x, output_name=output_name, sm=sm):
                    return self.model_low.run(x)[output_name] + sm.predict_values(np.atleast_2d(x))
        
            # Create m_k = lofi + RBF
            approximation_functions[output_name] = approximation_function
            
        self.approximation_functions = approximation_functions
        
    def plot_functions(self):
        n_plot = 9
        x_plot = np.linspace(self.bounds[0, 0], self.bounds[0, 1], n_plot)
        y_plot = np.linspace(self.bounds[1, 0], self.bounds[1, 1], n_plot)
        X, Y = np.meshgrid(x_plot, y_plot)
        x_values = np.vstack((X.flatten(), Y.flatten())).T
        
        y_plot_high = self.model_high.run_vec(x_values)[self.objective].reshape(n_plot, n_plot)
        
        # surrogate = []
        # for x_value in x_values:
        #     surrogate.append(np.squeeze(self.approximation_functions['con'](x_value)))
        # surrogate = np.array(surrogate)
        # y_plot_high = surrogate.reshape(n_plot, n_plot)
        
        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, y_plot_high, levels=101)
        plt.scatter(self.x[:, 0], self.x[:, 1], color='white')
        
        x = self.x[-1, 0]
        y = self.x[-1, 1]
        points = np.array([
            [x + self.trust_radius, y + self.trust_radius],
            [x + self.trust_radius, y - self.trust_radius],
            [x - self.trust_radius, y - self.trust_radius],
            [x - self.trust_radius, y + self.trust_radius],
            [x + self.trust_radius, y + self.trust_radius],
            ])
        plt.plot(points[:, 0], points[:, 1], 'w--')
        
        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])
        
        plt.xlabel('x0')
        plt.ylabel('x1')
        
        # plt.show()
        
        num_iter = self.x.shape[0]
        num_offset = 3
        
        if num_iter <= 5:
            for i in range(num_offset):
                plt.savefig(f'image_{self.counter}.png', dpi=300)
                self.counter += 1
        else:
            plt.savefig(f'image_{self.counter}.png', dpi=300)
            self.counter += 1
        
        
        
