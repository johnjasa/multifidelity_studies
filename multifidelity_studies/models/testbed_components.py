import numpy as np
import openmdao.api as om
from multifidelity_studies.models.base_model import BaseModel


A = 0.5
B = 10.
C = -5.

def simple_1D_high(x):
    term1 = A * (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    term2 = B * (x - 0.5)
    return term1 + term2 + C
    
def simple_1D_medium(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4) + B * (x - 0.5)
    
def simple_1D_low(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    

def simple_2D_high(x):
    term1 = A * (6 * x[1] - 2) ** 2
    term2 = B * (x[0] - 0.5)
    return term1 + term2 + C
    
def simple_2D_low(x):
    term1 = A * (4 * x[1] - 1) ** 2
    term2 = B * x[0]
    return term1 + term2
    
    
class simple_2D_high_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs['y'] = simple_2D_high(desvars['x'])
        outputs['con'] = np.sum(desvars['x'])
        return outputs

class simple_2D_low_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs['y'] = simple_2D_low(desvars['x'])
        outputs['con'] = np.sum(desvars['x']) + desvars['x'][1]**2 * outputs['y']
        return outputs
        

class simple_1D(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('fidelity', values=['low', 'high'])
        self.options.declare('n_pts')
    
    def setup(self):
        self.add_input('x', shape=self.options['n_pts'])
        
        self.add_output('y', shape=self.options['n_pts'])
        
        self.declare_partials('y', 'x', method='cs')
        
    def compute(self, inputs, outputs):
        fidelity = self.options['fidelity']
        x = inputs['x']

        if fidelity == 'high':
            outputs['y'] = simple_1D_high(x)
        else:
            outputs['y'] = simple_1D_low(x)