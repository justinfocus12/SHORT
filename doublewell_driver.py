import numpy as np
import time
from numpy import save,load
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0
font = {'family': 'serif', 'size': 25,}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import sys
import subprocess
import os
codefolder = "/home/jf4241/dgaf2"
os.chdir(codefolder)
from os import mkdir
from os.path import join,exists
from model_obj import Model
from doublewell_model import DoubleWellModel
import doublewell_params
import helper
from data_obj import Data
import function_obj 
from tpt_obj import TPT

# -----------------------
#    Make folders
datafolder = "/scratch/jf4241/SHORT_doublewell"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-05-30")
if not exists(dayfolder): mkdir(dayfolder)
expfolder = join(dayfolder,"1")
if not exists(expfolder): mkdir(expfolder)

asymb = r"$\mathbf{a}$"
bsymb = r"$\mathbf{b}$"

# ------------------------
# Decide what to do
least_action_flag = 0
run_long_flag =     0
run_short_flag =    0
compute_tpt_flag =  0
# ------------------------

# ------------------------
# Set parameters
algo_params,algo_param_string = doublewell_params.get_algo_params()
physical_params,physical_param_string = doublewell_params.get_physical_params()
# Set savefolder accordingly
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)
savefolder = join(physical_param_folder,algo_param_string)
if not exists(savefolder): mkdir(savefolder)
# ------------------------
# 1. Initialize the model
model = DoubleWellModel(physical_params)
# 2. Run long and short trajectories
long_simfolder,t_long,x_long = model.generate_data_long(simfolder,algo_params,run_long_flag=run_long_flag)
seed_weights = helper.reweight_data(x_long,model.sampling_features,model.sampling_density)

# -------------------------------------------
# 2 Find the least action pathway
if least_action_flag:
    model_lap = DoubleWellModel(physical_params)
    model_lap.minimize_action(50.0,physical_param_folder,dirn=-1,maxiter=100)
    model_lap.minimize_action(50.0,physical_param_folder,dirn=1,maxiter=100)
# Plot least action
model.plot_least_action(physical_param_folder)
# --------------------------
# Run short trajectories, either with single- or multiprocessor
short_simfolder = model.generate_data_short_multithreaded(x_long,simfolder,algo_params,seed_weights,run_short_flag=run_short_flag,overwrite_flag=False)

# ---------------------------------------------
# Initialize TPT
tpt = TPT(algo_params,physical_param_folder,long_simfolder,short_simfolder,savefolder)
# Initialize data
data = tpt.compile_data(model)
# Initialize function approximator as MSM basis
function = function_obj.MSMBasis(algo_params) #(basis_size,max_clust_per_level=max_clust_per_level,min_clust_size=min_clust_size)
# ----------------------------
# Computations
if compute_tpt_flag:
    tpt.label_x_long(model)
    tpt.compute_change_of_measure(model,data,function)
    tpt.compute_dam_moments_abba(model,data,function,num_moments=3) 
    tpt.compute_mfpt_unconditional(model,data,function)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
    tpt.write_compare_generalized_rates(model,data)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
#tpt.write_compare_generalized_rates(model,data)
#pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
# Computation: done
# -------------------------------------------

# -------------------------------------------
# Displays: starting
# -------------------------------------------
funlib = model.observable_function_library()

# ------------------------------------------
# Long plots
# 1D
long_fun = funlib["x0"]
tpt.plot_field_long(model,data,long_fun['fun'](data.X[:,0]),long_fun['name'],'x0',field_fun=long_fun['fun'],units=long_fun['units'],tmax=500)
# 2D
field_abbs = ["x0","x1"]
fieldnames = [funlib[f]["name"] for f in field_abbs]
field_funs = [funlib[f]["fun"] for f in field_abbs]
field_units = [funlib[f]["units"] for f in field_abbs]
field_unit_symbols = [funlib[f]["unit_symbol"] for f in field_abbs]
tpt.plot_field_long_2d(model,data,fieldnames,field_funs,field_abbs,units=field_units,tmax=500,field_unit_symbols=field_unit_symbols)
sys.exit()
# -------------------------------------------
# Casts and currents
theta_2d_abbs = [["x0","x1"]]
print("About to start displaying casts")
for i in range(len(theta_2d_abbs)):
    tpt.display_casts_abba(model,data,theta_2d_abbs[i:i+1])
    tpt.display_2d_currents(model,data,theta_2d_abbs[i:i+1])
sys.exit()


# Validation
theta_1d_fun = lambda x: x[:,:1]
theta_1d_name = r"$x_0$"
theta_1d_units = 1.0
theta_2d_fun = lambda x: x[:,:2]
theta_2d_names = [r"$x_0$",r"$x_1$"]
theta_2d_units = np.ones(2)
theta_2d_unit_symbols = ["",""]
theta_2d_abbs = ["x0","x1"]
tpt.display_change_of_measure_current(model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs)
tpt.display_change_of_measure_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
tpt.display_dam_moments_abba_current(model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs)
tpt.display_dam_moments_abba_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
sys.exit()
# ----------------------------
# Displays
# Load long trajectory to compare to reality
t_long,x_long = model.load_long_traj(long_simfolder)
x1_fun = lambda x: x[:,0]
tpt.plot_field_long(model,data,data.X[:,0,0],r"$x_1$","x1",field_fun=x1_fun,units=1.0)
theta_2d_short = data.X[:,0,:2]
theta_2d_long = x_long[:,:2] 
theta_1d_short = data.X[:,0,:1]
theta_1d_long = x_long[:,:1]
theta_1d_units = np.ones(1)
# Damage
tpt.display_dam_moments_ab(theta_1d_short,theta_1d_long,theta_2d_short,theta_2d_long,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
# Change of measure 
tpt.display_change_of_measure(theta_1d_short,theta_1d_long,theta_2d_short,theta_2d_long,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
# Committors
tpt.display_comm_ab(theta_1d_short,theta_1d_long,theta_2d_short,theta_2d_long,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
# Forward and backward MFPTs
# Regular method
tpt.display_mfpt_ab(theta_1d_short,theta_1d_long,theta_2d_short,theta_2d_long,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units,method=0)
# Now with the moments
tpt.display_mfpt_ab(theta_1d_short,theta_1d_long,theta_2d_short,theta_2d_long,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units,method=2)
# Conditional
tpt.display_conditional_mfpt_ab_moments(theta_1d_short,theta_1d_long,theta_2d_short,theta_2d_long,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)


