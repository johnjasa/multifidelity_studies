import os
import dill
from collections import OrderedDict
import numpy as np


class BaseModel():
    
    def __init__(self, desvars_init, warmstart_file):
        self.saved_desvars = []
        self.saved_outputs = []
        
        if os.path.exists(warmstart_file):
            with open(warmstart_file, 'rb') as f:
                saved_data = dill.load(f)
            self.saved_desvars = saved_data['desvars']
            self.saved_outputs = saved_data['outputs']
        
        self.warmstart_file = warmstart_file
        self.set_desvar_size_dict(desvars_init)
        
    def set_desvar_size_dict(self, desvars):
        self.desvar_sizes = OrderedDict()
        for key, value in desvars.items():
            self.desvar_sizes[key] = value.size
    
    def save_results(self, desvars, outputs):
        self.saved_desvars.append(desvars)
        self.saved_outputs.append(outputs)
        
        saved_data = {}
        saved_data['desvars'] = self.saved_desvars
        saved_data['outputs'] = self.saved_outputs
        with open(self.warmstart_file, 'wb') as f:
            dill.dump(saved_data, f)
        
    def load_results(self, desvars):
        for i, saved_dict in enumerate(self.saved_desvars):
            same_dict = True
            
            # Loop through the key/value pairs in each dictionary and see if they're
            # all the exact same. If one is not the same, they are not the same dict,
            # so we cannot use those saved results.
            for (key_1, value_1), (key_2, value_2) in zip(desvars.items(), saved_dict.items()):
                if not np.all(value_1 == value_2):
                    same_dict = False
                    break
                    
            if same_dict:
                return self.saved_outputs[i]
                
        return None
        
    def run_vec(self, list_of_desvars):
        list_of_results = []
        for i, desvars in enumerate(list_of_desvars):
            list_of_results.append(self.run(desvars))
        return np.squeeze(np.array(list_of_results))
        
    def flatten_desvars(self, desvars):
        flattened_desvars = []
        
        for key, value in desvars.items():
            flattened_value = np.flatten(value)
            flattened_desvars.append(flattened_value)
            
        return np.array(flattened_desvars)
        
    def unflatten_desvars(self, flattened_desvars):
        size_counter = 0
        desvars = OrderedDict()
        for key, size in self.desvar_sizes.items():
            desvars[key] = flattened_desvars[size_counter:size_counter+size]
            
        return desvars