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
from duffing_model import DuffingOscillator
import duffing_params
import helper
from data_obj import Data
import function_obj 
from tpt_obj import TPT

# ----------- Make folders ----------
datafolder = "/scratch/jf4241/SHORT_duffing"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-06-30")
if not exists(dayfolder): mkdir(dayfolder)
expfolder = join(dayfolder,"2")
if not exists(expfolder): mkdir(expfolder)
# -----------------------------------

asymb = r"$\mathbf{a}$"
bsymb = r"$\mathbf{b}$"

# ---------- Decide what to do ----------
least_action_flag = 1
run_long_flag =     1
run_short_flag =    1
compute_tpt_flag =  1
proj_1d_flag =      1
plot_long_2d_flag = 1
display_cast_flag = 1
lifecycle_flag =    1
gen_rates_flag =    1
plot_long_1d_flag = 1
validation_flag =   1
# Following flags only for HM model
regression_flag =   0
demo_flag =         0
qp_tb_coords_flag = 0
trans_state_flag =  0
# ---------------------------------------

# ---------- Set parameters --------------------------
algo_params,algo_param_string = duffing_params.get_algo_params()
physical_params,physical_param_string = duffing_params.get_physical_params()
# Set savefolder accordingly
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)
savefolder = join(physical_param_folder,algo_param_string)
if not exists(savefolder): mkdir(savefolder)
copyfile(join(codefolder,"duffing_params.py"),join(savefolder,"duffing_params.py"))
# -----------------------------------------------------
np.random.seed(0)
# ---------- 1. Initialize the model ----------
model = DuffingOscillator(physical_params,physical_param_string)
q = model.q # Dictionary of model parameters
# -------------------------------------------

# --------------- Long simulation --------------
print("About to run long trajectory")
long_simfolder,t_long,x_long = model.generate_data_long(simfolder,algo_params,run_long_flag=run_long_flag)
tmax = 1000.0
timax = np.where(t_long < tmax)[0][-1]
fig,ax = plt.subplots()
ax.plot(t_long[:timax],x_long[:timax,0])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
fig.savefig(join(savefolder,"xoft"))
plt.close(fig)
# Here's the trick: swap in a different x_seed
xvmax = np.array([2.0,0.3])
#xvmax = 1.1*np.max(np.abs(x_long),0)
bounds = np.array([[-xvmax[0],xvmax[0]],[-xvmax[1],xvmax[1]]])
shp = np.array([600,600])
x_seed = helper.both_grids(bounds,shp)[0]
print("x_seed: min={}, max={}".format(np.min(x_seed,0),np.max(x_seed,0)))
seed_weights = helper.reweight_data(x_seed,model.sampling_features,algo_params,model.sampling_density)
# ----------------------------------------------

# ---------- Short simulation ----------
print("About to run short trajectories")
short_simfolder = model.generate_data_short_multithreaded(x_seed,simfolder,algo_params,seed_weights,run_short_flag=run_short_flag,overwrite_flag=False)
# ---------------------------------------------

# ---------- 2. Find the least action pathway ----------
if least_action_flag:
    # Set the physical_params to be a small noise; optimized path should not depend on it
    physical_params['du_per_day'] = 1.0
    model_lap = DuffingOscillator(physical_params,physical_param_string)
    model_lap.minimize_action(10.0,physical_param_folder,dirn=-1,maxiter=10)
    model_lap.minimize_action(10.0,physical_param_folder,dirn=1,maxiter=10)
# Plot least action
#model.plot_least_action(physical_param_folder,"U")
# -------------------------------------------------------
# ---------- Initialize TPT ----------
tpt = TPT(algo_params,physical_param_folder,long_simfolder,short_simfolder,savefolder)
# Initialize data
data = tpt.compile_data(model)
# Initialize function approximator as MSM basis
function = function_obj.MSMBasis(algo_params)
# ---------------------------------------------

# ---------- Perform DGA ----------
if compute_tpt_flag:
    tpt.label_x_long(model)
    tpt.compute_change_of_measure(model,data,function)
    tpt.compute_dam_moments_abba(model,data,function)
    tpt.compute_mfpt_unconditional(model,data,function)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
    tpt.write_compare_generalized_rates(model,data)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
print("TPT computation: done")
# -------------------------------------------

tpt.label_x_long(model)

# --------- Reload data and prepare to plot -----------
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
print("Loaded TPT")
funlib = model.observable_function_library()
# -----------------------------------------------------


# ---------- Plot 1d committor proxies in an array ------
if proj_1d_flag:
    theta1d_list = ["x","v"]
    tpt.plot_projections_1d_array(model,data,theta1d_list)
# -------------------------------------------------------

# ---------- Demonstrate committor is better with 2 observables ------
if demo_flag:
    abblist = ["x","v"]
    def theta2d_fun(x):
        th = np.zeros((len(x),2))
        for i in range(2):
            th[:,i] = funlib[abblist[i]]["fun"](x).flatten()
        return th
    theta2d_names = [funlib[abb]["name"] for abb in abblist]
    theta2d_units = [funlib[abb]["units"] for abb in abblist]
    theta2d_unit_symbols = [funlib[abb]["unit_symbol"] for abb in abblist]
    tpt.demonstrate_committor_mfpt(model,data,theta2d_fun,theta2d_names,theta2d_units,theta2d_unit_symbols)
# -----------------------------------------------------------------

# ---------- Forecast probability and lead time as independent variables -------------------
if qp_tb_coords_flag:
    tpt.plot_prediction_curves_colored(model,data)
# -----------------------------------------------------------------

# ---------- Plot dominant transition states-----------
if trans_state_flag:
    tpt.plot_transition_states(model,data)
# -------------------------------------------------

# ------------- Plot long trajectory in 2D -----------
if plot_long_2d_flag:
    field_abbs = ["x","v"]
    fieldnames = [funlib[f]["name"] for f in field_abbs]
    field_funs = [funlib[f]["fun"] for f in field_abbs]
    field_units = [funlib[f]["units"] for f in field_abbs]
    field_unit_symbols = [funlib[f]["unit_symbol"] for f in field_abbs]
    tpt.plot_field_long_2d(model,data,fieldnames,field_funs,field_abbs,units=field_units,tmax=3000,field_unit_symbols=field_unit_symbols)
# ----------------------------------------------------

# ----------- Display casts and currents in 2d -----------
if display_cast_flag:
    theta_2d_abbs = [["x","v"]]
    print("About to start displaying casts")
    for i in range(len(theta_2d_abbs)):
        tpt.display_casts_abba(model,data,theta_2d_abbs[i:i+1])
    tpt.display_2d_currents(model,data,theta_2d_abbs)
# --------------------------------------------------------

# ----------- Display lifecycle correlations -----------
if lifecycle_flag:
    tpt.write_compare_lifecycle_correlations(model,data)
    tpt.plot_lifecycle_correlations_bar(model)
# ------------------------------------------------------

# ----------- Write, plot, and validate generalized rates ------
if gen_rates_flag:
    tpt.write_compare_generalized_rates(model,data)
# --------------------------------------------------------------

# ----------- Plot long trajectory in 1d ---------------
if plot_long_1d_flag:
    field_fun = funlib["x"]
    tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"x",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
    field_fun = funlib["v"]
    tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"v",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
# ------------------------------------------------------


# ------------------- Validation -----------------------
if validation_flag:
    val_2d_names = ["x","v"]
    val_1d_name = "x"
    q = model.q
    #n = q['Nz']-1
    theta_1d_fun = lambda x: funlib[val_1d_name]["fun"](x).reshape(-1,1)
    theta_1d_name = funlib[val_1d_name]["name"]
    theta_1d_units = funlib[val_1d_name]["units"]
    theta_2d_fun = lambda x: model.sampling_features(x,algo_params) #cv_sample_fun
    theta_2d_names = algo_params['sampling_feature_names']
    theta_2d_units = np.array([funlib[name]['units'] for name in theta_2d_names]) #np.array([q['length']**2/q['time'],q['length']/q['time']])
    theta_2d_unit_symbol = [funlib[name]['unit_symbol'] for name in theta_2d_names] #[r"$m^2/s$",r"$m/s$"]
    tpt.display_change_of_measure_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
    tpt.display_dam_moments_abba_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
# ----------------------------------------------------
