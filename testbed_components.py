import numpy as np
import openmdao.api as om


class simple_one_D(om.ExplicitComponent):
    
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
        A = 0.5
        B = 10.
        C = -5.
        
        if fidelity == 'high':
            term1 = A * (6 * x - 2) ** 2 * np.sin(12 * x - 4)
            term2 = B * (x - 0.5)
            outputs['y'] = term1 + term2 + C
            
        else:
            outputs['y'] = (6 * x - 2) ** 2 * np.sin(12 * x - 4)