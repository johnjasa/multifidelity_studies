# base method class
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
import smt.surrogate_models as smt


class BaseMethod():
    
    def __init__(self, model_low, model_high):
        
        self.list_of_desvars = []
        self.initialize_points()
        
        desvars = self.list_of_desvars[-1]
        
        self.model_low = model_low(desvars, 'low_2D_results.pkl')
        self.model_high = model_high(desvars, 'high_2D_results.pkl')
        
    def construct_approximation(self, interp_method='smt'):
        y_low = self.model_low.run_vec(self.list_of_desvars)
        y_high = self.model_high.run_vec(self.list_of_desvars)
        differences = y_high - y_low
        
        if interp_method == 'rbf':
            input_arrays = np.split(self.x, self.x.shape[1], axis=1)
            input_arrays = [x.flatten() for x in input_arrays]
            
            # Construct RBF interpolater for error function
            e = Rbf(*input_arrays, differences)
            
            # Create m_k = lofi + RBF
            def approximation_function(x):
                desvars = self.model_low.unflatten_desvars(x)
                return self.model_low.run(desvars) + e(*x)
                
        elif interp_method == 'smt':
            xt = self.x
            yt = differences
            
            xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])

            sm = smt.RMTB(
                xlimits=xlimits,
                order=3,
                num_ctrl_pts=3,
                print_global=False,
            )
            sm.set_training_values(xt, yt)
            sm.train()
            
            def approximation_function(x):
                desvars = self.model_low.unflatten_desvars(x)
                return self.model_low.run(desvars) + sm.predict_values(np.atleast_2d(x))
        
        # Create m_k = lofi + RBF
        self.approximation_function = approximation_function
        
    def initialize_points(self, num_initial_points=5):
        self.n_dims = 2
        
        x_init = np.random.rand(num_initial_points, self.n_dims)
        x = x_init.copy()
        
        for i in range(num_initial_points):
            desvars = OrderedDict()
            desvars['x'] = x[i, :]
            self.list_of_desvars.append(desvars)
            
        self.x = x
        
    def plot_functions(self):
        n_plot = 11
        x_plot = np.linspace(0., 1., n_plot)
        X, Y = np.meshgrid(x_plot, x_plot)
        x_values = np.vstack((X.flatten(), Y.flatten())).T
        
        list_of_desvars_plot = []
        surrogate = np.zeros((n_plot*n_plot))
        for j in range(n_plot*n_plot):
            desvars = OrderedDict()
            desvars['x'] = x_values[j]
            list_of_desvars_plot.append(desvars)
            surrogate[j] = self.approximation_function(x_values[j])
            
        y_plot_high = self.model_high.run_vec(list_of_desvars_plot).reshape(n_plot, n_plot)
        # y_plot_high = surrogate.reshape(n_plot, n_plot)
    
        plt.figure()
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
        
        plt.xlim([0., 1])
        plt.ylim([0., 1])
        
        plt.show()
        
        
