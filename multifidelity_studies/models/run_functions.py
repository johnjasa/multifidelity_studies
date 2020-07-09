import os
from collections import OrderedDict
import numpy as np
from wisdem.glue_code.runWISDEM import run_wisdem
from wisdem.commonse.mpi_tools import MPI
from multifidelity_studies.models.base_model import BaseModel


## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
fname_wt_input = run_dir + "IEA-15-240-RWT_WISDEMieaontology4all.yaml"
fname_analysis_options_ccblade = run_dir + "analysis_options_ccblade.yaml"
fname_analysis_options_openfast = run_dir + "analysis_options_openfast.yaml"
fname_opt_options = run_dir + "optimization_options.yaml"
folder_output = run_dir + "it_0/"
fname_wt_output = folder_output + "temp.yaml"


class CCBlade(BaseModel):
    
    def run(self, desvars):
        
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
            
            outputs = wt_opt_ccblade["ccblade.CP"][0]
            
            self.save_results(desvars, outputs)
            
            return outputs
            
        else:
            return loaded_results

class OpenFAST(BaseModel):
        
    def run(self, desvars):
        
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