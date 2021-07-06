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

# ----------- Make folders ----------
datafolder = "/scratch/jf4241/SHORT_holtonmass"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-07-04")
if not exists(dayfolder): mkdir(dayfolder)
expfolder = join(dayfolder,"0")
if not exists(expfolder): mkdir(expfolder)
# -----------------------------------

asymb = r"$\mathbf{a}$"
bsymb = r"$\mathbf{b}$"

# ---------- Decide what to do ----------
least_action_flag = 0
run_long_flag =     0
run_short_flag =    0
compute_tpt_flag =  0
regression_flag =   0
proj_1d_flag =      0
demo_flag =         0
qp_tb_coords_flag = 0
trans_state_flag =  1
plot_long_2d_flag = 0
display_cast_flag = 0
lifecycle_flag =    0
gen_rates_flag =    0
plot_long_1d_flag = 0
validation_flag =   0
# ---------------------------------------

# ---------- Set parameters --------------------------
algo_params,algo_param_string = hm_params.get_algo_params()
physical_params,physical_param_string = hm_params.get_physical_params()
# Set savefolder accordingly
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)
savefolder = join(physical_param_folder,algo_param_string)
if not exists(savefolder): mkdir(savefolder)
copyfile(join(codefolder,"hm_params.py"),join(savefolder,"hm_params.py"))
# -----------------------------------------------------
np.random.seed(0)
# ---------- 1. Initialize the model ----------
model = HoltonMassModel(physical_params)
q = model.q # Dictionary of model parameters
print("q['Gsq'] = {}".format(q['Gsq']))
fig,ax = model.plot_two_snapshots(model.xst[0],model.xst[1],asymb,bsymb)
fig.savefig(join(savefolder,"snapshots_AB"))
plt.close(fig)
print("Done plotting snapshots")
# -------------------------------------------

# ---------- 2. Find the least action pathway ----------
if least_action_flag:
    # Set the physical_params to be a small noise; optimized path should not depend on it
    physical_params['du_per_day'] = 1.0
    model_lap = HoltonMassModel(physical_params)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=-1,maxiter=10)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=1,maxiter=10)
# Plot least action
model.plot_least_action(physical_param_folder)
#sys.exit()
#model.plot_least_action(physical_param_folder,"U")
#model.plot_least_action(physical_param_folder,"mag")
#model.plot_least_action(physical_param_folder,"vTint")
# -------------------------------------------------------

# --------------- Long simulation --------------
print("About to run long trajectory")
long_simfolder,t_long,x_long = model.generate_data_long(simfolder,algo_params,run_long_flag=run_long_flag)
seed_weights = helper.reweight_data(x_long,model.sampling_features,algo_params,model.sampling_density)
# ----------------------------------------------

# ---------- Short simulation ----------
print("About to run short trajectories")
short_simfolder = model.generate_data_short_multithreaded(x_long,simfolder,algo_params,seed_weights,run_short_flag=run_short_flag,overwrite_flag=False)
# ---------------------------------------------

# ---------- Initialize TPT ----------
tpt = TPT(algo_params,physical_param_folder,long_simfolder,short_simfolder,savefolder)
# Initialize data
data = tpt.compile_data(model,istart=algo_params["istart"])
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

# --------- Reload data and prepare to plot -----------
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
print("Loaded TPT")
funlib = model.observable_function_library()
# -----------------------------------------------------

# ---------- Regress forward committor --------
if regression_flag:
    lasso_fun = lambda x: model.regression_features(x)
    num_regression_features = 6
    tpt.beta_allz_qp,tpt.intercept_allz_qp,tpt.score_allz_qp,tpt.recon_allz_qp = tpt.regress_committor_modular(model,data,lasso_fun,method='LASSO')
    model.plot_sparse_regression_allz(tpt.beta_allz_qp,tpt.score_allz_qp,tpt.savefolder,suffix="_qp")
    # Regression restricted to one level at a time
    z = q['z_d'][1:-1]/1000
    tpt.beta_onez = np.zeros((len(z),num_regression_features))
    tpt.intercept_onez = np.zeros(len(z))
    tpt.score_onez = np.zeros(len(z))
    lasso_x = lasso_fun(data.X[:,0])
    for i in range(len(z)):
        lasso_fun_z = lambda x: model.regression_features(x)[:,[j*(q['Nz']-1)+i for j in range(num_regression_features)]]
        tpt.beta_onez[i],tpt.intercept_onez[i],tpt.score_onez[i],_ = tpt.regress_committor_modular(model,data,lasso_fun_z,method='LASSO')
    model.plot_sparse_regression_zslices(tpt.beta_onez,tpt.score_onez,tpt.savefolder,suffix="_qp")
    # Plot the family of 1-dimensional committor proxies parameterized by z (altitude)
    tpt.plot_zfam_committor(model,data)
# -------------------------------------------------------

# ---------- Plot 1d committor proxies in an array ------
if proj_1d_flag:
    theta1d_list = ['Uref','vTintref']
    tpt.plot_projections_1d_array(model,data,theta1d_list)
# -------------------------------------------------------

# ---------- Demonstrate committor is better with 2 observables ------
if demo_flag:
    abblist = ["vTintref","Uref"]
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


# ------------- Plot long trajectory in 2D -----------
if plot_long_2d_flag:
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
# ----------------------------------------------------

# ----------- Display casts and currents in 2d -----------
if display_cast_flag:
    theta_2d_abbs = [["magref","Uref"],["vTintref","Uref"]]
    print("About to start displaying casts")
    for i in range(len(theta_2d_abbs)):
        tpt.display_casts_abba(model,data,theta_2d_abbs[i:i+1])
    tpt.display_2d_currents(model,data,theta_2d_abbs)
# --------------------------------------------------------

# ----------- Display lifecycle correlations -----------
if lifecycle_flag:
    keys=['Uref_ln20','heatflux_g3em5','vTintref_l0']
    tpt.write_compare_lifecycle_correlations(model,data)
    tpt.plot_lifecycle_correlations_bar(model,keys=keys)
# ------------------------------------------------------

# ----------- Write, plot, and validate generalized rates ------
if gen_rates_flag:
    tpt.write_compare_generalized_rates(model,data)
# --------------------------------------------------------------

# ----------- Plot long trajectory in 1d ---------------
if plot_long_1d_flag:
    field_fun = funlib["vTintref"]
    tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"vTintref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
    field_fun = funlib["vTref"]
    tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"vTref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
    field_fun = funlib["Uref"]
    tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"Uref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
    field_fun = funlib["magref"]
    tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"magref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"])
# ------------------------------------------------------

# ---------- Plot dominant transition states-----------
if trans_state_flag:
    tpt.plot_transition_states_all(model,data,collect_flag=False)
    #tpt.plot_transition_states_committor(model,data,preload_idx=True)
    #tpt.plot_transition_states_leadtime(model,data,preload_idx=True)
# -------------------------------------------------

# ------------------- Validation -----------------------
if validation_flag:
    val_2d_names = ["magref","Uref"]
    val_1d_name = "Uref"
    q = model.q
    n = q['Nz']-1
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
