import os
import dill
from collections import OrderedDict
import numpy as np


class BaseModel():
    
    def __init__(self, desvars_init, warmstart_file=None):
        self.saved_desvars = []
        self.saved_outputs = []
        
        self.warmstart_file = warmstart_file
        if warmstart_file is not None:
            if os.path.exists(warmstart_file):
                with open(warmstart_file, 'rb') as f:
                    saved_data = dill.load(f)
                self.saved_desvars = saved_data['desvars']
                self.saved_outputs = saved_data['outputs']
                
        self.set_desvar_size_dict(desvars_init)
        
    def set_desvar_size_dict(self, desvars):
        self.desvar_sizes = OrderedDict()
        total_size = 0
        for key, value in desvars.items():
            self.desvar_sizes[key] = value.size
            total_size += value.size
        self.total_size = total_size
    
    def save_results(self, desvars, outputs):
        self.saved_desvars.append(self.flatten_desvars(desvars))
        self.saved_outputs.append(outputs)
        
        if self.warmstart_file is not None:
            saved_data = {}
            saved_data['desvars'] = self.saved_desvars
            saved_data['outputs'] = self.saved_outputs
            with open(self.warmstart_file, 'wb') as f:
                dill.dump(saved_data, f)
        
    def load_results(self, flattened_desvars):
        for i, saved_desvar in enumerate(self.saved_desvars):
            same_inputs = True
            
            if not np.all(flattened_desvars == saved_desvar):
                same_inputs = False
                continue
                    
            if same_inputs:
                return self.saved_outputs[i]
                
        return None
        
    def run(self, flattened_desvars):
        loaded_results = self.load_results(flattened_desvars)
        if loaded_results is None:
            desvars = self.unflatten_desvars(flattened_desvars)
            outputs = self.compute(desvars)
            self.save_results(desvars, outputs)
            return outputs
        else:
            return loaded_results
        
    def run_vec(self, x):
        dict_of_results = {}
        for i, flattened_desvars in enumerate(x):
            outputs = self.run(flattened_desvars)
            
            for key in outputs:
                if key not in dict_of_results:
                    dict_of_results[key] = []
                dict_of_results[key].append(outputs[key])
                
        for key in dict_of_results:
            dict_of_results[key] = np.squeeze(np.array(dict_of_results[key]))
            
        return dict_of_results
        
    def flatten_desvars(self, desvars):
        flattened_desvars = []
        
        for key, value in desvars.items():
            flattened_value = np.squeeze(value)
            flattened_desvars.extend(flattened_value)
            
        return np.array(flattened_desvars)
        
    def unflatten_desvars(self, flattened_desvars):
        size_counter = 0
        desvars = OrderedDict()
        for key, size in self.desvar_sizes.items():
            desvars[key] = flattened_desvars[size_counter:size_counter+size]
            
        return desvars