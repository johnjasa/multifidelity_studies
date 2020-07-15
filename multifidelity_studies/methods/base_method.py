# base method class
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
import smt.surrogate_models as smt


class BaseMethod():
    
    def __init__(self, model_low, model_high, bounds, num_initial_points=5):
        
        self.bounds = np.array(bounds)
        
        self.model_low = model_low
        self.model_high = model_high
        
        self.initialize_points(num_initial_points)
        self.counter = 0
        
    def construct_approximation(self, interp_method='smt'):
        y_low = self.model_low.run_vec(self.x)
        y_high = self.model_high.run_vec(self.x)
        differences = y_high - y_low
        
        if interp_method == 'rbf':
            input_arrays = np.split(self.x, self.x.shape[1], axis=1)
            input_arrays = [x.flatten() for x in input_arrays]
            
            # Construct RBF interpolater for error function
            e = Rbf(*input_arrays, differences)
            
            # Create m_k = lofi + RBF
            def approximation_function(x):
                return self.model_low.run(x) + e(*x)
                
        elif interp_method == 'smt':
            xt = self.x
            yt = differences
            
            sm = smt.RMTB(
                xlimits=self.bounds,
                order=3,
                num_ctrl_pts=3,
                print_global=False,
            )
            sm.set_training_values(xt, yt)
            sm.train()
            
            def approximation_function(x):
                return self.model_low.run(x) + sm.predict_values(np.atleast_2d(x))
        
        # Create m_k = lofi + RBF
        self.approximation_function = approximation_function
        
    def initialize_points(self, num_initial_points):
        self.n_dims = self.model_high.total_size
        x_init_raw = np.random.rand(num_initial_points, self.n_dims)
        self.x = x_init_raw * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        
    def plot_functions(self):
        n_plot = 11
        x_plot = np.linspace(0., 1., n_plot)
        X, Y = np.meshgrid(x_plot, x_plot)
        x_values = np.vstack((X.flatten(), Y.flatten())).T
        
        y_plot_high = self.model_high.run_vec(x_values).reshape(n_plot, n_plot)
    
        plt.figure(figsize=(6,4))
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
        
        num_iter = self.x.shape[0]
        num_offset = 10
        
        if num_iter <= 5:
            for i in range(num_offset):
                plt.savefig(f'image_{self.counter}.png', dpi=300)
                self.counter += 1
        else:
            plt.savefig(f'image_{self.counter}.png', dpi=300)
            self.counter += 1
                
        
        
