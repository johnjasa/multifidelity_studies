from wisdem.glue_code.runWISDEM import run_wisdem
from wisdem.commonse.mpi_tools import MPI
import os


## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
# fname_wt_input         = run_dir + 'IEAonshoreWT_3.yaml'
fname_wt_input = run_dir + "IEA-15-240-RWT_WISDEMieaontology4all.yaml"
fname_analysis_options_ccblade = run_dir + "analysis_options_ccblade.yaml"
fname_analysis_options_openfast = run_dir + "analysis_options_openfast.yaml"
fname_opt_options = run_dir + "optimization_options.yaml"
folder_output = run_dir + "it_0/"
fname_wt_output = folder_output + "temp.yaml"


def run_ccblade():
    wt_opt_ccblade, analysis_options_ccblade, opt_options_ccblade = run_wisdem(
        fname_wt_input,
        fname_analysis_options_ccblade,
        fname_opt_options,
        fname_wt_output,
        folder_output,
    )
    return wt_opt_ccblade["ccblade.CP"]


def run_openfast():
    wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_wisdem(
        fname_wt_input,
        fname_analysis_options_openfast,
        fname_opt_options,
        fname_wt_output,
        folder_output,
    )
    return wt_opt_openfast["aeroelastic.Cp"]

print(run_openfast())
print(run_ccblade())