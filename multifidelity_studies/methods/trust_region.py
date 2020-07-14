# base method class
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
import smt.surrogate_models as smt
from multifidelity_studies.models.testbed_components import simple_2D_high_model, simple_2D_low_model
from multifidelity_studies.methods.base_method import BaseMethod


class SimpleTrustRegion(BaseMethod):
    
    def __init__(self, model_low, model_high, max_trust_radius=1000., eta=0.15, gtol=1e-4, trust_radius=0.2):
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
        res = minimize(self.approximation_function, x0, method='SLSQP', tol=1e-10, bounds=bounds)
        x_new = res.x
        
        if np.any(np.abs(lower_bounds - x_new) < 1e-6) or np.any(np.abs(upper_bounds - x_new) < 1e-6):
            hits_boundary = True
        else:
            hits_boundary = False
            
        return x_new, hits_boundary
    
    def update_trust_region(self, x_new, hits_boundary):
        # 3. Compute the ratio of actual improvement to predicted improvement
        desvars = OrderedDict()
        desvars['x'] = np.squeeze(x_new)
        
        actual_reduction = self.model_high.run(self.list_of_desvars[-1]) - self.model_high.run(desvars)
        predicted_reduction = self.model_high.run(self.list_of_desvars[-1]) - self.approximation_function(x_new)
        
        # 4. Accept or reject the trial point according to that ratio
        if predicted_reduction <= 0:
            print('not enough reduction! rejecting point')
        else:
            self.x = np.vstack((self.x, np.atleast_2d(x_new)))
            self.list_of_desvars.append(desvars)
            
        if predicted_reduction == 0.:
            rho = 0.
        else:
            rho = actual_reduction / predicted_reduction
    
        # 5. Update trust region according to rho_k
        eta = 0.25
        if rho < eta:
            self.trust_radius *= 0.25
        elif rho > eta and hits_boundary:
            self.trust_radius = min(2*self.trust_radius, self.max_trust_radius)
        print('trust radius', self.trust_radius)
            
    def optimize(self):
        self.construct_approximation()
        # self.plot_functions()
        
        for i in range(20):
            x_new, hits_boundary = self.find_next_point()
            
            self.update_trust_region(x_new, hits_boundary)
                
            print(self.x)
            print(self.model_high.run_vec(self.list_of_desvars))
            print()
            
            self.construct_approximation()
        
            # self.plot_functions()
            
            if self.trust_radius <= 1e-4:
                print("Found optimal point!")
                break
        

# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

trust_region = SimpleTrustRegion(simple_2D_low_model, simple_2D_high_model)
trust_region.optimize()

