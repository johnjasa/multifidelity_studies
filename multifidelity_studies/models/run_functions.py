import os
from collections import OrderedDict
import numpy as np
from wisdem.glue_code.runWISDEM import run_wisdem
from wisdem.commonse.mpi_tools import MPI
from multifidelity_studies.models.base_model import BaseModel
from scipy.interpolate import PchipInterpolator
import dill
from wisdem.ccblade import CCBlade as CCBladeOrig


## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
fname_wt_input = run_dir + "IEA-15-240-RWT_WISDEMieaontology4all.yaml"
fname_analysis_options_ccblade = run_dir + "analysis_options_ccblade.yaml"
fname_analysis_options_openfast = run_dir + "analysis_options_openfast.yaml"
fname_opt_options = run_dir + "optimization_options.yaml"
folder_output = run_dir + "it_0/"
fname_wt_output = folder_output + "temp.yaml"


class FullCCBlade(BaseModel):
    def compute(self, desvars):
        wt_opt_ccblade, analysis_options_ccblade, opt_options_ccblade = run_wisdem(
            fname_wt_input,
            fname_analysis_options_ccblade,
            fname_opt_options,
            fname_wt_output,
            folder_output,
            desvars,
        )

        outputs = {}
        outputs["CP"] = wt_opt_ccblade["ccblade.CP"][0]

        return outputs


class OpenFAST(BaseModel):
    def compute(self, desvars):
        wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_wisdem(
            fname_wt_input,
            fname_analysis_options_openfast,
            fname_opt_options,
            fname_wt_output,
            folder_output,
            desvars,
        )

        outputs = {}
        outputs["CP"] = wt_opt_openfast["aeroelastic.Cp"][0]

        return outputs


class CCBlade(BaseModel):
    def compute(self, desvars):
        
        with open('CCBlade_inputs.pkl', 'rb') as f:
            saved_dict = dill.load(f)
        
        chord_opt_gain = desvars['blade.opt_var.chord_opt_gain']

        chord_original = np.array([5.2       , 5.22814651, 5.32105729, 5.45815745, 5.60229318,
        5.71827031, 5.76707382, 5.71284486, 5.53647944, 5.29094421,
        5.03450401, 4.81470288, 4.62321904, 4.43204464, 4.24515221,
        4.06546765, 3.89602837, 3.73506076, 3.57914991, 3.42460161,
        3.26800581, 3.11191777, 2.95713751, 2.79989844, 2.6365298 ,
        2.46395353, 2.28349782, 2.09593192, 1.90186155, 0.5       ])
        s_opt_chord = np.linspace(0., 1., len(chord_opt_gain))
        s = np.array([0.        , 0.03448276, 0.06896552, 0.10344828, 0.13793103,
        0.17241379, 0.20689655, 0.24137931, 0.27586207, 0.31034483,
        0.34482759, 0.37931034, 0.4137931 , 0.44827586, 0.48275862,
        0.51724138, 0.55172414, 0.5862069 , 0.62068966, 0.65517241,
        0.68965517, 0.72413793, 0.75862069, 0.79310345, 0.82758621,
        0.86206897, 0.89655172, 0.93103448, 0.96551724, 1.        ])
        
        spline         = PchipInterpolator
        chord_spline            = spline(s_opt_chord, chord_opt_gain)
        chord  = chord_original * chord_spline(s)
        
        get_cp_cm = CCBladeOrig(saved_dict['r'],
            chord,
            saved_dict['twist'],
            saved_dict['af'],
            saved_dict['Rhub'],
            saved_dict['Rtip'],
            saved_dict['nBlades'],
            saved_dict['rho'],
            saved_dict['mu'],
            saved_dict['precone'],
            saved_dict['tilt'],
            saved_dict['yaw'],
            saved_dict['shearExp'],
            saved_dict['hub_height'],
            saved_dict['nSector'],
            saved_dict['precurve'],
            saved_dict['precurveTip'],
            saved_dict['presweep'],
            saved_dict['presweepTip'],
            saved_dict['tiploss'],
            saved_dict['hubloss'],
            saved_dict['wakerotation'],
            saved_dict['usecd'],
            )   
        get_cp_cm.inverse_analysis = False
        get_cp_cm.induction        = True

        # Compute omega given TSR
        Omega   = saved_dict['Uhub']*saved_dict['tsr']/saved_dict['Rtip'] * 30.0/np.pi

        myout, derivs = get_cp_cm.evaluate([saved_dict['Uhub']], [Omega], [saved_dict['pitch']], coefficients=True)
        
        outputs = {}
        outputs["CP"] = myout['CP']

        return outputs