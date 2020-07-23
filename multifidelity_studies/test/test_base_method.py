import unittest
import numpy as np
from multifidelity_studies.models.testbed_components import simple_2D_high_model, simple_2D_low_model
from multifidelity_studies.methods.base_method import BaseMethod


class Test(unittest.TestCase):

    def test_set_initial_point(self):
        np.random.seed(13)
        
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        desvars = {'x' : np.array([0., 0.25])}
        model_low = simple_2D_low_model(desvars)
        model_high = simple_2D_high_model(desvars)
        method_instance = BaseMethod(model_low, model_high, bounds, disp=False)
        
        method_instance.add_objective('y')
        method_instance.set_initial_point([0.5, 0.5])
        
        np.testing.assert_allclose(method_instance.x[-1, :], [0.5, 0.5], )
        
    def test_bounds_and_initial_points(self):
        np.random.seed(13)
        
        bounds = np.array([[-10., 11.0], [-20.0, 1.0]])
        desvars = {'x' : np.array([0., 0.25])}
        model_low = simple_2D_low_model(desvars)
        model_high = simple_2D_high_model(desvars)
        method_instance = BaseMethod(model_low, model_high, bounds, disp=False)
        
        init_points = np.array([[  6.33175062, -15.01163438],
                                [  7.30984919,   0.28073316],
                                [ 10.42462339, -10.4775658 ],
                                [  2.78989172,  -3.71394319],
                                [  3.47388024,  -4.83761718]])
        
        np.testing.assert_allclose(method_instance.x, init_points)
        
        
        np.random.seed(13)
        
        method_instance = BaseMethod(model_low, model_high, bounds, disp=False, num_initial_points=3)
        
        np.testing.assert_allclose(method_instance.x, init_points[:3, :])
        
    def test_approximation(self):
        np.random.seed(13)
        
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        desvars = {'x' : np.array([0., 0.25])}
        model_low = simple_2D_low_model(desvars)
        model_high = simple_2D_high_model(desvars)
        method_instance = BaseMethod(model_low, model_high, bounds, disp=False)
        
        method_instance.add_objective('y')
        method_instance.construct_approximations()
        
        func = method_instance.approximation_functions['y']
        
        flattened_desvars = model_low.flatten_desvars(desvars)
        np.testing.assert_allclose(func(flattened_desvars), -5.33064616)
        

if __name__ == '__main__':
    unittest.main()