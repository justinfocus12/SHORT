import numpy as np
import matplotlib.pyplot as plt
from hm_model import HoltonMassModel
import hm_params 

physical_params,physical_param_string = hm_params.get_physical_params()
xst = np.load("xst.npy")
model = HoltonMassModel(physical_params,xst)
#np.save("xst",model.xst)

afrac = np.linspace(0,1,4)
bfrac = 1 - afrac
x = np.outer(afrac, model.xst[0]) + np.outer(bfrac, model.xst[1])
model.test_enstrophy_equation(x,0.0001)

