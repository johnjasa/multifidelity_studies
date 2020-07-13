# base method class
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
from multifidelity_studies.models.testbed_components import simple_2D_high_model, simple_2D_low_model


class BaseMethod():
    
    def __init__(self, model_low, model_high):
        
        self.list_of_desvars = []
        self.initialize_points()
        
        desvars = self.list_of_desvars[-1]
        
        self.model_low = simple_2D_low_model(desvars, 'low_2D_results.pkl')
        self.model_high = simple_2D_high_model(desvars, 'high_2D_results.pkl')
        
    def construct_approximation(self):
        y_low = self.model_low.run_vec(self.list_of_desvars)
        y_high = self.model_high.run_vec(self.list_of_desvars)
        
        input_arrays = np.split(self.x, self.x.shape[1], axis=1)
        
        input_arrays = [x.flatten() for x in input_arrays]
        
        # Construct RBF interpolater for error function
        differences = y_high - y_low
        e = Rbf(*input_arrays, differences)
        
        # Create m_k = lofi + RBF
        def approximation_function(x):
            desvars = self.model_low.unflatten_desvars(x)
            return self.model_low.run(desvars) + e(*x)
        
        # Create m_k = lofi + RBF
        self.approximation_function = approximation_function
        
    def initialize_points(self, num_initial_points=3):
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
        for j in range(n_plot*n_plot):
            desvars = OrderedDict()
            desvars['x'] = x_values[j]
            list_of_desvars_plot.append(desvars)
            
        y_plot_high = self.model_high.run_vec(list_of_desvars_plot).reshape(n_plot, n_plot)
    
        plt.figure()
        plt.contourf(X, Y, y_plot_high)
        plt.scatter(self.x[:, 0], self.x[:, 1], color='white')
        plt.show()
        
        
class SimpleTrustRegion(BaseMethod):
    
    def __init__(self, model_low, model_high, max_trust_radius=1000., eta=0.15, gtol=1e-4, trust_radius=0.1):
        super().__init__(model_low, model_high)
        
        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.gtol = gtol
        self.trust_radius = trust_radius
        
    def find_next_point(self):
        x0 = self.x[-1, :]
        
        # min (m_k(x_k + s_k)) st ||x_k|| <= del K
        lower_bounds = x0 - self.trust_radius
        lower_bounds[lower_bounds < 0.] = 0.
        upper_bounds = x0 + self.trust_radius
        upper_bounds[upper_bounds > 1.] = 1.
        
        bounds = list(zip(lower_bounds, upper_bounds))
        res = minimize(self.approximation_function, x0, method='SLSQP', tol=1e-6, bounds=bounds)
        x_new = np.atleast_2d(res.x)
        self.x = np.vstack((self.x, x_new))
        
        desvars = OrderedDict()
        desvars['x'] = np.squeeze(x_new)
        self.list_of_desvars.append(desvars)
        
        if np.any(np.abs(lower_bounds - x_new) < 1e-6) or np.any(np.abs(upper_bounds - x_new) < 1e-6):
            hits_boundary = True
        else:
            hits_boundary = False
            
        return x_new, hits_boundary
    
    def update_trust_region(self, x_new, hits_boundary):
        # 3. Compute the ratio of actual improvement to predicted improvement
        
        actual_reduction = self.model_high.run(self.list_of_desvars[-2]) - self.model_high.run(self.list_of_desvars[-1])
        predicted_reduction = self.model_high.run(self.list_of_desvars[-2]) - self.approximation_function(self.x[-1, :])
        
        # 4. Accept or reject the trial point according to that ratio
        if predicted_reduction <= 0:
            print('not enough reduction! probably at a local optimum')
            fail_flag = True
        else:
            fail_flag = False
            
        rho = actual_reduction / predicted_reduction
    
        # 5. Update trust region according to rho_k
        if rho < 0.25:
            self.trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            self.trust_radius = min(2*self.trust_radius, self.max_trust_radius)
            
        return fail_flag
        
    def optimize(self):
        self.construct_approximation()
        
        for i in range(10):
            x_new, hits_boundary = self.find_next_point()
            
            fail_flag = self.update_trust_region(x_new, hits_boundary)
            
            if fail_flag:
                break
                
            self.construct_approximation()
        
            print(self.x)
        
            self.plot_functions()
        

# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

trust_region = SimpleTrustRegion(simple_2D_low_model, simple_2D_high_model)
trust_region.optimize()

