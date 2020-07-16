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
    
    def __init__(self, model_low, model_high, bounds, max_trust_radius=1000., eta=0.15, gtol=1e-4, trust_radius=0.2):
        super().__init__(model_low, model_high, bounds)
        
        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.gtol = gtol
        self.trust_radius = trust_radius
        
    def process_constraints(self):
        list_of_constraints = []
        for constraint in self.constraints:
            scipy_constraint = {}
            
            func = self.approximation_functions[constraint['name']]
            if constraint['equals'] is not None:
                scipy_constraint['type'] = 'eq'
                scipy_constraint['fun'] = lambda x: np.squeeze(func(x) - constraint['equals'])
                
            if constraint['upper'] is not None:
                scipy_constraint['type'] = 'ineq'
                scipy_constraint['fun'] = lambda x: np.squeeze(constraint['upper'] - func(x))
                
            if constraint['lower'] is not None:
                scipy_constraint['type'] = 'ineq'
                scipy_constraint['fun'] = lambda x: np.squeeze(func(x) - constraint['lower'])
                
            list_of_constraints.append(scipy_constraint)
            
        self.list_of_constraints = list_of_constraints
            
        
    def find_next_point(self):
        x0 = self.x[-1, :]
        
        # min (m_k(x_k + s_k)) st ||x_k|| <= del K
        trust_region_lower_bounds = x0 - self.trust_radius
        lower_bounds = np.maximum(trust_region_lower_bounds, self.bounds[:, 0])
        trust_region_upper_bounds = x0 + self.trust_radius
        upper_bounds = np.minimum(trust_region_upper_bounds, self.bounds[:, 1])
        
        bounds = list(zip(lower_bounds, upper_bounds))
        scaled_function = lambda x: self.objective_scaler * self.approximation_functions[self.objective](x)
        res = minimize(scaled_function, x0, method='SLSQP', tol=1e-10, bounds=bounds, constraints=self.list_of_constraints, options={'disp':False})
        x_new = res.x
        
        tol = 1e-6
        if np.any(np.abs(trust_region_lower_bounds - x_new) < tol) or np.any(np.abs(trust_region_upper_bounds - x_new) < tol):
            hits_boundary = True
        else:
            hits_boundary = False
            
        return x_new, hits_boundary
    
    def update_trust_region(self, x_new, hits_boundary):
        # 3. Compute the ratio of actual improvement to predicted improvement
        prev_point_high = self.objective_scaler * self.model_high.run(self.x[-1])[self.objective]
        new_point_high = self.objective_scaler * self.model_high.run(x_new)[self.objective]
        new_point_approx = self.objective_scaler * self.approximation_functions[self.objective](x_new)
        
        actual_reduction = prev_point_high - new_point_high
        predicted_reduction = prev_point_high - new_point_approx
        
        # 4. Accept or reject the trial point according to that ratio
        # Unclear if this logic is needed; it's better to update the surrogate model with a bad point, even
        if predicted_reduction <= 0:
            print('not enough reduction! rejecting point')
        else:
            self.x = np.vstack((self.x, np.atleast_2d(x_new)))
            
        if predicted_reduction == 0.:
            rho = 0.
        else:
            rho = actual_reduction / predicted_reduction
    
        # 5. Update trust region according to rho_k
        eta = 0.25
        if rho >= eta or hits_boundary:
            self.trust_radius = min(2*self.trust_radius, self.max_trust_radius)
        elif rho < eta:  # Unclear if this is the best check
            self.trust_radius *= 0.25
        print('trust radius', self.trust_radius)
            
    def optimize(self, plot=False):
        self.construct_approximations()
        self.process_constraints()
        
        if plot:
            self.plot_functions()
        
        for i in range(30):
            self.process_constraints()
            x_new, hits_boundary = self.find_next_point()
            
            self.update_trust_region(x_new, hits_boundary)
                
            self.construct_approximations()
        
            if plot:
                self.plot_functions()
                
            x_test = self.x[-1, :]
            
            if self.trust_radius <= 1e-6:
                break
                
        print()
        print("Found optimal point!")
        print(self.x[-1, :])
        print(self.model_high.run(self.x[-1, :])[self.objective])
        
