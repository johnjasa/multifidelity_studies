# base method class
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from multifidelity_studies.models.testbed_components import simple_1D_low, simple_1D_high


class BaseMethod():
    
    def __init__(self, func_low, func_high):
        
        self.func_low = func_low
        self.func_high = func_high
        
        self.initialize_points()
        
    def construct_approximation(self, x):
        y_low = self.func_low(x)
        y_high = self.func_high(x)
        
        # Construct RBF interpolater for error function
        differences = y_high - y_low
        e = Rbf(x, differences)
        
        # Create m_k = lofi + RBF
        self.approximation_function = lambda x: simple_1D_low(x) + e(x)
        
    def initialize_points(self, num_initial_points=3):
        # Query hifi and lofi in same places
        x = np.random.rand(num_initial_points)
        y_low = self.func_low(x)
        y_high = self.func_high(x)
        self.x = x
        
    # Misc utilities
    def plot_functions(self):
        x_plot = np.linspace(0., 1., 101)
        surrogate = self.approximation_function(x_plot)
        y_plot_low = self.func_low(x_plot)
        y_plot_high = self.func_high(x_plot)

        plt.figure()
        plt.plot(x_plot, y_plot_low, label='low')
        plt.plot(x_plot, y_plot_high, label='high')
        plt.plot(x_plot, surrogate, label='mixed surrogate')
        plt.scatter(self.x, self.func_high(self.x), color='k')
        plt.legend()
        plt.show()  
        
        
class SimpleTrustRegion(BaseMethod):
    
    def __init__(self, func_low, func_high, max_trust_radius=1000., eta=0.15, gtol=1e-4, trust_radius=0.1):
        super().__init__(func_low, func_high)
        
        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.gtol = gtol
        self.trust_radius = trust_radius
        
    def find_next_point(self):
        x0 = self.x[-1]
        
        # min (m_k(x_k + s_k)) st ||x_k|| <= del K
        bounds = [(max(x0 - self.trust_radius, 0.), min(1., x0 + self.trust_radius))]
        res = minimize(self.approximation_function, x0, method='SLSQP', tol=1e-6, bounds=bounds)
        x_new = res.x
        self.x = np.hstack((self.x, x_new))
        
        bounds = np.array(bounds)
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        
        if np.any(np.abs(lower_bounds - x_new) < 1e-6) or np.any(np.abs(upper_bounds - x_new) < 1e-6):
            hits_boundary = True
        else:
            hits_boundary = False
            
        return x_new, hits_boundary
    
    def update_trust_region(self, x_new, hits_boundary):
        # 3. Compute the ratio of actual improvement to predicted improvement
        actual_reduction = self.func_high(self.x[-2]) - self.func_high(x_new)
        predicted_reduction = self.func_high(self.x[-2]) - self.approximation_function(x_new)
        
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
        self.construct_approximation(self.x)
        
        for i in range(10):
            x_new, hits_boundary = self.find_next_point()
            
            fail_flag = self.update_trust_region(x_new, hits_boundary)
            
            if fail_flag:
                break
                
            self.construct_approximation(self.x)
        
            print(self.x)
        
            self.plot_functions()
        

# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(3)

trust_region = SimpleTrustRegion(simple_1D_low, simple_1D_high)
trust_region.optimize()

