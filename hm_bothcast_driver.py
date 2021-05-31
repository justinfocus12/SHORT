import numpy as np
import time
from numpy import save,load
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 25
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
from os import mkdir
from os.path import join,exists
from shutil import copyfile
codefolder = "/home/jf4241/SHORT"
os.chdir(codefolder)
from model_obj import Model
from hm_model import HoltonMassModel
import hm_params
import helper
from data_obj import Data
import function_obj 
from tpt_obj import TPT

# ---------------------------
# Make folders
datafolder = "/scratch/jf4241/SHORT_holtonmass"
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
run_short_flag =    1
compute_tpt_flag =  1
# ------------------------

# ------------------------
# Set parameters
algo_params,algo_param_string = hm_params.get_algo_params()
physical_params,physical_param_string = hm_params.get_physical_params()
# Set savefolder accordingly
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)
savefolder = join(physical_param_folder,algo_param_string)
if not exists(savefolder): mkdir(savefolder)
copyfile(join(codefolder,"hm_params.py"),join(savefolder,"hm_params.py"))
# ---------------------
np.random.seed(0)
#-------------------------------
# 1. Initialize the model
model = HoltonMassModel(physical_params)
fig,ax = model.plot_two_snapshots(model.xst[0],model.xst[1],"A","B")
fig.savefig(join(savefolder,"snapshots_AB"))
plt.close(fig)
print("Done plotting snapshots")
# -------------------------------------------

# -------------------------------------------
# 2. Find the least action pathway
if least_action_flag:
    model_lap = HoltonMassModel(physical_params)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=-1,maxiter=10)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=1,maxiter=10)
# Plot least action
model.plot_least_action(physical_param_folder,"U")
model.plot_least_action(physical_param_folder,"mag")
model.plot_least_action(physical_param_folder,"vTint")
# --------------------------------------------

# ----------------------------------------------
# Long simulation
long_simfolder,t_long,x_long = model.generate_data_long(simfolder,algo_params,run_long_flag=run_long_flag)
seed_weights = helper.reweight_data(x_long,model.sampling_features,model.sampling_density)
# ----------------------------------------------

# ---------------------------------------------
# Run short trajectories
print("About to run short trajectories")
short_simfolder = model.generate_data_short_multithreaded(x_long,simfolder,algo_params,seed_weights,run_short_flag=run_short_flag,overwrite_flag=False)
# ---------------------------------------------

#---------------------------------------------
# Initialize TPT
tpt = TPT(algo_params,physical_param_folder,long_simfolder,short_simfolder,savefolder)
# Initialize data
data = tpt.compile_data(model)
# Initialize function approximator as MSM basis
function = function_obj.MSMBasis(algo_params)
# ---------------------------------------------

# ---------------------------------------------
# Perform DGA
if compute_tpt_flag:
    tpt.label_x_long(model)
    tpt.compute_change_of_measure(model,data,function)
    tpt.compute_dam_moments_abba(model,data,function,num_moments=4)
    tpt.compute_mfpt_unconditional(model,data,function)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
    tpt.write_compare_generalized_rates(model,data)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
#tpt.write_compare_generalized_rates(model,data)
#pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
# Computation: done
# -------------------------------------------

# ---------------------------------------------
# Starting displays
print("Loaded TPT")
tpt.plot_transition_states(model,data)
sys.exit()
# Plot long trajectory in 2D
field_abbs = ["magref","Uref"]
fieldnames = [funlib[f]["name"] for f in field_abbs]
field_funs = [funlib[f]["fun"] for f in field_abbs]
field_units = [funlib[f]["units"] for f in field_abbs]
field_unit_symbols = [funlib[f]["unit_symbol"] for f in field_abbs]
tpt.plot_field_long_2d(model,data,fieldnames,field_funs,field_abbs,units=field_units,tmax=3000,field_unit_symbols=field_unit_symbols)
field_abbs = ["vTintref","Uref"]
fieldnames = [funlib[f]["name"] for f in field_abbs]
field_funs = [funlib[f]["fun"] for f in field_abbs]
field_units = [funlib[f]["units"] for f in field_abbs]
field_unit_symbols = [funlib[f]["unit_symbol"] for f in field_abbs]
tpt.plot_field_long_2d(model,data,fieldnames,field_funs,field_abbs,units=field_units,tmax=3000,field_unit_symbols=field_unit_symbols)
sys.exit()
# 2D casts and currents
theta_2d_abbs = [["magref","Uref"],["vTintref","Uref"]]
print("About to start displaying casts")
for i in range(len(theta_2d_abbs)):
    tpt.display_casts_abba(model,data,theta_2d_abbs[i:i+1])
tpt.lag_time_current_display = lag_time_current_display
tpt.display_2d_currents(model,data,theta_2d_abbs)
sys.exit()
keys=['Uref_ln20','magref_g1e7','heatflux_g5em5']
tpt.write_compare_lifecycle_correlations(model,data)
tpt.plot_lifecycle_correlations(model,keys=keys)
sys.exit()
tpt.write_compare_generalized_rates(model,data)
sys.exit()
# Plot the long trajectory in 1D
field_fun = funlib["vTintref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"vTintref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
field_fun = funlib["vTref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"vTref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
field_fun = funlib["Uref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"Uref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
field_fun = funlib["magref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"magref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
#sys.exit()
#theta_2d_abbs = [["U19","U67"],["mag19","U19"],["magref","Uref"],["repsiref","impsiref"],["dqdymean","q2mean"]]
#sys.exit()
# integral distributions
#tpt.write_compare_generalized_rates(model,data)
#pdf2png(1)
# Regression
lasso_fun = lambda x: x
tpt.regress_committor_modular(model,data,lasso_fun,method='LASSO')
model.plot_sparse_regression(tpt.lasso_beta,tpt.lasso_score,tpt.savefolder)

# Validation
theta_1d_fun = lambda x: x[:,2*n+q['zi']:2*n+q['zi']+1]
theta_1d_name = r"$U(30 km)$"
theta_1d_units = q['length']/q['time']
theta_2d_fun = cv_sample_fun
theta_2d_names = [r"$|\Psi(30 km)|$",r"$U(30 km)$"]
theta_2d_units = np.array([q['length']**2/q['time'],q['length']/q['time']])
theta_2d_unit_symbols = [r"$m^2/s$",r"$m/s$"]
tpt.display_change_of_measure_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
tpt.display_dam_moments_abba_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
#pdf2png(0)
