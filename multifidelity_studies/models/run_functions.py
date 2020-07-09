import os
from collections import OrderedDict
import numpy as np
import dill
from wisdem.glue_code.runWISDEM import run_wisdem
from wisdem.commonse.mpi_tools import MPI


## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
fname_wt_input = run_dir + "IEA-15-240-RWT_WISDEMieaontology4all.yaml"
fname_analysis_options_ccblade = run_dir + "analysis_options_ccblade.yaml"
fname_analysis_options_openfast = run_dir + "analysis_options_openfast.yaml"
fname_opt_options = run_dir + "optimization_options.yaml"
folder_output = run_dir + "it_0/"
fname_wt_output = folder_output + "temp.yaml"


class BaseModel():
    
    def __init__(self, warmstart_file):
        self.saved_desvars = []
        self.saved_outputs = []
        
        if os.path.exists(warmstart_file):
            with open(warmstart_file, 'rb') as f:
                saved_data = dill.load(f)
            self.saved_desvars = saved_data['desvars']
            self.saved_outputs = saved_data['outputs']
        
        self.warmstart_file = warmstart_file
    
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
                print('Loaded saved results!')
                return self.saved_outputs[i]
                
        return None
        
    def run_vec(self, chords):
        results = np.zeros(len(chords))
        for i, chord in enumerate(chords):
            results[i] = self.run(chord)
        return results

class CCBlade(BaseModel):
    
    def run(self, chord):
        desvars = OrderedDict()
        desvars['blade.opt_var.chord_opt_gain'] = chord
        
        loaded_results = self.load_results(desvars)
        if loaded_results is None:
            wt_opt_ccblade, analysis_options_ccblade, opt_options_ccblade = run_wisdem(
                fname_wt_input,
                fname_analysis_options_ccblade,
                fname_opt_options,
                fname_wt_output,
                folder_output,
                desvars,
            )
            
            outputs = wt_opt_ccblade["ccblade.CP"]
            
            self.save_results(desvars, outputs)
            
            return outputs
            
        else:
            return loaded_results

class OpenFAST(BaseModel):
        
    def run(self, chord):
        desvars = OrderedDict()
        desvars['blade.opt_var.chord_opt_gain'] = chord
        
        loaded_results = self.load_results(desvars)
        if loaded_results is None:
            wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_wisdem(
                fname_wt_input,
                fname_analysis_options_openfast,
                fname_opt_options,
                fname_wt_output,
                folder_output,
                desvars,
            )
            
            outputs = wt_opt_openfast["aeroelastic.Cp"][0]
            
            self.save_results(desvars, outputs)
            
            return outputs
            
        else:
            return loaded_results