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
codefolder = "/home/jf4241/dgaf2"
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
datafolder = "/scratch/jf4241/dgaf2_holtonmass"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-04-07")
if not exists(dayfolder): mkdir(dayfolder)
expfolder = join(dayfolder,"radb30")
if not exists(expfolder): mkdir(expfolder)

# ------------------------
# Decide what to do
least_action_flag = bool(0)
run_long_flag =     bool(0)
run_short_flag =    bool(0)
compute_tpt_flag =  bool(0)
# ------------------------

# ------------------------
# Set parameters
# Physical parameters
du_per_day = 1.0
dh_per_day = 0.0
hB_d = 38.5
adef_dim =               75
bdef_dim =               75
bdef_thresh =            "beta" # or zero or beta
ref_alt =                30.0 # 21.5 or 26.9 or 29.6
physical_param_string = ("du{}_h{}_ad{}_bd{}".format(du_per_day,hB_d,adef_dim,bdef_dim)).replace(".","p")
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)
print("phys param folder = {}".format(physical_param_folder))
# Algorithmic parameters
tmax_long = 500000.0
tmax_short = 20.0
dt_save = 0.5
nshort = 500000 #500000 #200000
basis_type = 'MSM'
basis_size = 1500 #200
lag_time = 20.0 #0.2
nlags = 21 #21
lag_time_seq = np.linspace(0,lag_time,nlags)
print("lag_time_seq = {}".format(lag_time_seq))
lag_time_current = lag_time_seq[1]
lag_time_current_display = lag_time_seq[4]
min_clust_size = 5
if min_clust_size*basis_size > nshort:
    sys.exit("basis_size*min_clust_size > nshort")
# Set savefolder accordingly
algo_param_string = ("tlong{}_N{}_bs{}_mcs{}_lag{}_nlags{}_lagj{}".format(tmax_long,nshort,basis_size,min_clust_size,lag_time,nlags,lag_time_current)).replace('.','p')
savefolder = join(physical_param_folder,algo_param_string)
if not exists(savefolder): mkdir(savefolder)
copyfile(join(codefolder,"pdf2png.sh"),join(savefolder,"pdf2png.sh"))
# ------------------------
def pdf2png(exit_flag=1):
    os.chdir(savefolder)
    os.system("source ./pdf2png.sh")
    os.chdir(codefolder)
    if exit_flag: sys.exit()
    return

# ---------------------
np.random.seed(0)
#-------------------------------
# 1. Specify the model
model = HoltonMassModel(hB_d=hB_d,du_per_day=du_per_day,dh_per_day=dh_per_day,ref_alt=ref_alt)
print("model.dam_dict.keys() = {}".format(model.dam_dict.keys()))
fig,ax = model.plot_two_snapshots(model.xst[0],model.xst[1],"A","B")
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
function = function_obj.MSMBasis(basis_size,max_clust_per_level=100,min_clust_size=10)

if compute_tpt_flag:
    # Define the TPT object
    tpt.label_x_long(model)
    tpt.compute_change_of_measure(model,data,function)
    tpt.compute_dam_moments_abba(model,data,function,num_moments=4)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
    tpt.write_compare_generalized_rates(model,data)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
np.random.seed(1)
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
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
#pdf2png(0)
sys.exit()

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
