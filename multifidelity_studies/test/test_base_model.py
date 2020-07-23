import unittest
import numpy as np
from multifidelity_studies.models.base_model import BaseModel


class Test(unittest.TestCase):

    def test_float_desvars(self):
        desvars_init = {}
        desvars_init['x1'] = 1.5
        desvars_init['x2'] = 2.2
        
        model = BaseModel(desvars_init)
        
        self.assertEqual(model.total_size, 2)

    def test_error_msg(self):
        desvars_init = {}
        desvars_init['x1'] = 1.5
        
        model = BaseModel(desvars_init)
        
        with self.assertRaises(NotImplementedError) as raises_msg:
            model.compute()
            
        exception = raises_msg.exception
        msg =  "This method needs to be provided by the inheriting model class."
        self.assertEqual(exception.args[0], msg)
        
if __name__ == '__main__':
    unittest.main()