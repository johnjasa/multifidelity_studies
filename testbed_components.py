import numpy as np
import openmdao.api as om


def simple_1D_high(x):
    A = 0.5
    B = 10.
    C = -5.
    
    term1 = A * (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    term2 = B * (x - 0.5)
    return term1 + term2 + C
    
def simple_1D_low(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


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