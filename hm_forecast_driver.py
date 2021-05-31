# This is for paper 1 results
import numpy as np
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
codefolder = "/home/jf4241/SHORT"
os.chdir(codefolder)
from model_obj import Model
from hm_model import HoltonMassModel
import helper
from data_obj import Data
from basis_obj import Basis
import function_obj 
from tpt_obj import TPT
from shutil import copyfile

# ---------------------------
# Make folders
datafolder = "/scratch/jf4241/SHORT_holtonmass"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-05-29")
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

# Algorithmic parameters
tmax_long = 500000.0
tmax_short = 20.0
dt_save = 0.5
nshort = 500000 
basis_type = 'MSM'
basis_size = 1500 
lag_time = 20.0 
nlags = int(lag_time) + 1 
lag_time_seq = np.linspace(0,lag_time,nlags)
lag_time_current = lag_time_seq[1]  # For computing rates
lag_time_current_display = lag_time_seq[4] # For displaying current vector fields
max_clust_per_level = 100
min_clust_size = 10
if min_clust_size*basis_size > nshort:
    sys.exit("basis_size*min_clust_size > nshort")

# Physical parameters
du_per_day = 1.0
dh_per_day = 0.0
hB_d = 38.5
ref_alt =                30.0 # 21.5 or 26.9 or 29.6
physical_param_string = ("du{}_h{}".format(du_per_day,hB_d)).replace(".","p")
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)

# Set savefolder accordingly
algo_param_string = ("tlong{}_N{}_bs{}_mcs{}_lag{}_nlags{}_lagj{}".format(tmax_long,nshort,basis_size,min_clust_size,lag_time,nlags,lag_time_current)).replace('.','p')
savefolder = join(physical_param_folder,algo_param_string)
if not exists(savefolder): mkdir(savefolder)

np.random.seed(0)

#-------------------------------
# 1. Specify the model
model = HoltonMassModel(hB_d=hB_d,du_per_day=du_per_day,dh_per_day=dh_per_day,ref_alt=ref_alt,abdefdim=abdefdim)
print("model.dam_dict.keys() = {}".format(model.dam_dict.keys()))
fig,ax = model.plot_two_snapshots(model.xst[0],model.xst[1],asymb,bsymb)
fig.savefig(join(savefolder,"snapshots_AB"))
plt.close(fig)
print("Done plotting snapshots")
#pdf2png(0)

# 2. Create the data
# Long simulation
long_simfolder,t_long,x_long = model.generate_data_long(simfolder,tmax_long,dt_save,run_long_flag=run_long_flag)
#pdf2png(0)

# 2.5 Find the least action pathway
if least_action_flag:
    model_lap = HoltonMassModel(hB_d=hB_d,du_per_day=1.0,dh_per_day=dh_per_day,ref_alt=ref_alt)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=-1,maxiter=10)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=1,maxiter=10)
# Plot least action
model.plot_least_action(physical_param_folder,"U")
model.plot_least_action(physical_param_folder,"mag")
model.plot_least_action(physical_param_folder,"vTint")

# Resample short data from long
funlib = model.observable_function_library()
q = model.q
n = q['Nz']-1
def cv_sample_fun(x):
    Nx,xdim = x.shape
    magu = np.zeros((Nx,2))
    #magu[:,0] = x[:,q['zi']]
    #magu[:,1] = x[:,n+q['zi']]
    #magu[:,2] = x[:,2*n+q['zi']]
    magu[:,0] = np.sqrt(x[:,q['zi']]**2 + x[:,n+q['zi']]**2)
    magu[:,1] = x[:,2*n+q['zi']]
    return magu
cv_sample_dim = 2
cv_sample_pdf = lambda x: np.ones(len(x))
seed_weights = helper.reweight_data(x_long,cv_sample_fun,cv_sample_pdf)
# Now the short data
short_simfolder = model.generate_data_short_multithreaded(x_long,simfolder,tmax_short,nshort,dt_save,seed_weights,run_short_flag=run_short_flag,overwrite_flag=False)
#--------------------------
# Initialize TPT
tpt = TPT(nshort,lag_time_current,lag_time_seq,physical_param_folder,long_simfolder,short_simfolder,savefolder,lag_time_current_display)
data = tpt.compile_data(model)
print("Finished loading data")
function = function_obj.MSMBasis(basis_size,max_clust_per_level=max_clust_per_level,min_clust_size=min_clust_size)


if compute_tpt_flag:
# Define the TPT object
    tpt.label_x_long(model)
    tpt.compute_change_of_measure(model,data,function)
    tpt.compute_dam_moments_abba(model,data,function,num_moments=2)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
    # Compute MFPT
    tpt.compute_mfpt_unconditional(model,data,function)
    tpt.write_compare_generalized_rates(model,data)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
np.random.seed(1)
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
print("Loaded TPT")
# ---------------------------------
# Regression on committor
# Regression with all levels at once
lasso_fun = lambda x: model.regression_features(x)
num_regression_features = 6
tpt.beta_allz,tpt.intercept_allz,tpt.score_allz,tpt.recon_allz = tpt.regress_committor_modular(model,data,lasso_fun,method='LASSO')
model.plot_sparse_regression_allz(tpt.beta_allz,tpt.score_allz,tpt.savefolder,suffix="_qp")
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
# z family
tpt.plot_zfam(model,data)
sys.exit()
# ---------------------------------
# ---------------------------------
# Regression on lead time
# All levels at once
lasso_fun = lambda x: model.regression_features(x)
num_regression_features = 6
tpt.beta_allz_tb,tpt.intercept_allz_tb,tpt.score_allz_tb,tpt.recon_allz_tb = tpt.regress_leadtime_modular(model,data,lasso_fun,method='LASSO')
model.plot_sparse_regression_allz(tpt.beta_allz_tb,tpt.score_allz_tb,tpt.savefolder,suffix="_tb")
# Regression restricted to one level at a time
z = q['z_d'][1:-1]/1000
tpt.beta_onez_tb = np.zeros((len(z),num_regression_features))
tpt.intercept_onez_tb = np.zeros(len(z))
tpt.score_onez_tb = np.zeros(len(z))
lasso_x = lasso_fun(data.X[:,0])
for i in range(len(z)):
    lasso_fun_z = lambda x: model.regression_features(x)[:,[j*(q['Nz']-1)+i for j in range(num_regression_features)]]
    tpt.beta_onez_tb[i],tpt.intercept_onez_tb[i],tpt.score_onez_tb[i],_ = tpt.regress_leadtime_modular(model,data,lasso_fun_z,method='LASSO')
model.plot_sparse_regression_zslices(tpt.beta_onez_tb,tpt.score_onez_tb,tpt.savefolder,suffix="_tb")
# ---------------------------------
sys.exit()
# ---------------------------------
# 1d plots
tpt.plot_projections_1d_array(model,data)
#sys.exit()
# ---------------------------------
# ---------------------------------
# The money plot
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
#sys.exit()
# ---------------------------------
# ------------------------------------
# Forecast probability vs. lead time 
#tpt.plot_prediction_curves() 
tpt.plot_prediction_curves_colored(model,data)
sys.exit()
# ------------------------------------
# --------------------------------------------------------
#   Plot the transition states as defined by current
tpt.plot_transition_states(model,data)
sys.exit()
# -------------------------------------------------------
# --------------------------------
sys.exit()
# ---------------------------------
# Compare sim times
# First, against U
theta_1d_abb = "Uref"
theta_1d_fun = funlib["Uref"]["fun"]
theta_1d_name = funlib["Uref"]["name"]
theta_1d_units = funlib["Uref"]["units"]
theta_1d_unit_symbol = funlib["Uref"]["unit_symbol"]
#theta_1d_fun = lambda x: x[:,2*n+q['zi']:2*n+q['zi']+1]
#theta_1d_name = r"$U$(30 km)"
#theta_1d_units = q['length']/q['time']
#theta_1d_unit_symbol = 
tpt.display_equiv_sim_time(model,data,theta_1d_name,theta_1d_units,theta_1d_unit_symbol,basis_size,theta_1d_abb,theta_1d_fun=theta_1d_fun)
# Second, against q+
#theta_1d_name = r"$q^+$"
#theta_1d_units = 1.0
#theta_1d_abb = "qp"
#theta_1d_short = tpt.dam_moments['one']['xb'][0,:,0]
#tpt.display_equiv_sim_time(model,data,theta_1d_name,theta_1d_units,basis_size,theta_1d_abb,theta_1d_short=theta_1d_short)
#sys.exit()
# ---------------------------------
# ---------------------------
# Numerical validation
# Error in subspaces
theta_1d_fun = lambda x: x[:,2*n+q['zi']:2*n+q['zi']+1]
theta_1d_name = r"$U(30 km)$"
theta_1d_units = q['length']/q['time']
theta_2d_fun = cv_sample_fun
theta_2d_names = [r"$|\Psi(30 km)|$",r"$U(30 km)$"]
theta_2d_units = np.array([q['length']**2/q['time'],q['length']/q['time']])
theta_2d_unit_symbols = [r"$m^2/s$",r"$m/s$"]
tpt.display_dam_moments_abba_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
tpt.display_change_of_measure_validation(model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
#sys.exit() 
# Advantage w.r.t. direct numerical simulation
#theta_1d_fun = lambda x: x[:,2*n+q['zi']:2*n+q['zi']+1]
#theta_1d_name = r"$U(30 km)$"
#theta_1d_units = q['length']/q['time']
#theta_1d_unit_symbol = r"$m/s$"
#tpt.compute_naive_time(model,data,theta_1d_fun,theta_1d_name,theta_1d_units,theta_1d_unit_symbol)
#sys.exit()
# --------------------------------

# ---------------------------------------------
# Long plot 
# 1d: U
field_fun = funlib["Uref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"Uref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"],include_reactive=False)
# 2d: IHF and U
field_abbs = ["vTintref","Uref"]
fieldnames = [funlib[f]["name"] for f in field_abbs]
field_funs = [funlib[f]["fun"] for f in field_abbs]
field_units = [funlib[f]["units"] for f in field_abbs]
field_unit_symbols = [funlib[f]["unit_symbol"] for f in field_abbs]
tpt.plot_field_long_2d(model,data,fieldnames,field_funs,field_abbs,units=field_units,tmax=3000,field_unit_symbols=field_unit_symbols)
#sys.exit()
# ------------------------------------------





sys.exit()







# Ensemble plot of lead times
fig4_theta_list = ["Uref","vTintref"] #,"LASSO"]
tid_list = np.array([5])
for i in range(len(tid_list)):
    trans_id = tid_list[i]
    tpt.plot_transition_ensemble_multiple(model,data,fig4_theta_list,trans_id)
#sys.exit()

#--------------------------------
# 2D casts and currents
theta_2d_abbs = [["mag21p5","U13p5"],["impsi21p5","U21p5"],["vTint21p5","U13p5"]]
print("About to start displaying casts")
for i in range(len(theta_2d_abbs)):
    tpt.display_casts_abba(model,data,theta_2d_abbs[i:i+1])
#sys.exit()
tpt.lag_time_current_display = lag_time_current_display
tpt.display_2d_currents(model,data,theta_2d_abbs)
sys.exit()
#--------------------------------










# Plot the long trajectory in 1D
field_fun = funlib["vTintref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"vTintref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"],include_reactive=False)
field_fun = funlib["vTref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"vTref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"],include_reactive=False)
field_fun = funlib["magref"]
tpt.plot_field_long(model,data,field_fun["fun"](data.X[:,0]),field_fun["name"],"magref",field_fun=field_fun["fun"],units=field_fun["units"],tmax=3000,time_unit_symbol="days",field_unit_symbol=field_fun["unit_symbol"],include_reactive=False)
#sys.exit()










# Plot long trajectory in 2D
field_abbs = ["magref","Uref"]
fieldnames = [funlib[f]["name"] for f in field_abbs]
field_funs = [funlib[f]["fun"] for f in field_abbs]
field_units = [funlib[f]["units"] for f in field_abbs]
field_unit_symbols = [funlib[f]["unit_symbol"] for f in field_abbs]
tpt.plot_field_long_2d(model,data,fieldnames,field_funs,field_abbs,units=field_units,tmax=3000,field_unit_symbols=field_unit_symbols)
#sys.exit()
#keys=['Uref_ln20','magref_g1e7','heatflux_g5em5']
#tpt.write_compare_lifecycle_correlations(model,data)
#tpt.plot_lifecycle_correlations(model,keys=keys)
#sys.exit()
tpt.write_compare_generalized_rates(model,data)
#sys.exit()
#theta_2d_abbs = [["U19","U67"],["mag19","U19"],["magref","Uref"],["repsiref","impsiref"],["dqdymean","q2mean"]]
#sys.exit()
# integral distributions
#tpt.write_compare_generalized_rates(model,data)
#pdf2png(1)
#pdf2png(0)
#sys.exit()

#pdf2png(0)
