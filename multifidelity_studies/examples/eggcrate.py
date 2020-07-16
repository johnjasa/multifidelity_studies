import numpy as np
from multifidelity_studies.models.testbed_components import eggcrate_high_model, eggcrate_low_model
from multifidelity_studies.methods.trust_region import SimpleTrustRegion


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
desvars = {'x' : np.array([0., 0.])}
model_low = eggcrate_low_model(desvars)
model_high = eggcrate_high_model(desvars)
trust_region = SimpleTrustRegion(model_low, model_high, bounds, trust_radius=3)

trust_region.add_objective('y')

trust_region.optimize(plot=False)
