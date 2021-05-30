import numpy as np
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
from oscillating_doublewell_model import OscillatingDoubleWellModel
import helper
from data_obj import Data
from basis_obj import Basis
import function_obj 
from tpt_obj import TPT
from shutil import copyfile

# -----------------------
#    Make folders
datafolder = "/scratch/jf4241/dgaf2_oscillating_doublewell"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-03-09")
if not exists(dayfolder): mkdir(dayfolder)
expfolder = join(dayfolder,"1")
if not exists(expfolder): mkdir(expfolder)

# ------------------------
# Decide what to do
run_long_flag =    bool(0)
run_short_flag =   bool(1)
compute_tpt_flag = bool(1)
# ------------------------

# ------------------------
# Set parameters
# Physical parameters
tau = 0.25
kappa = 0.0
lam = 0.5
sigma = 1.0
physical_param_string = ("tau{}_kappa{}_lam{}_sigma{}".format(tau,kappa,lam,sigma)).replace(".","p")
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)
# Algorithmic parameters
tmax_long = 20000.0
tmax_short = 0.4
dt_save = 0.001
nshort = 200000 #200000
basis_size = 200 #200
lag_time = 0.1 #0.2
nlags = 6 #21
lag_time_seq = np.linspace(0,lag_time,nlags)
print("lag_time_seq = {}".format(lag_time_seq))
lag_time_current = lag_time_seq[-1]
# Set savefolder accordingly
algo_param_string = ("tlong{}_N{}_bs{}_lag{}_nlags{}_lagj{}".format(tmax_long,nshort,basis_size,lag_time,nlags,lag_time_current)).replace('.','p')
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

# -----------------------
# 1. Initialize the model
state_dim = 2
odw = OscillatingDoubleWellModel(state_dim,tau=tau,kappa=kappa,lam=lam,sigma=sigma)
#xst_approx = odw.approximate_fixed_points()
#odw.find_fixed_points(tmax=10)
# ----------------------

# ----------------------------------------------
# New: alternate sampling with TPT
# Initialize first
theta_fun = lambda x: x
theta_pdf = lambda x: np.ones(len(x))
long_simfolder,t_long,x_long = odw.generate_data_long(simfolder,tmax_long,dt_save,run_long_flag=run_long_flag)

# Run short trajectories
seed_weights = helper.reweight_data(x_long,theta_fun,theta_pdf) 
short_simfolder = odw.generate_data_short(x_long,simfolder,tmax_short,nshort,dt_save,seed_weights,run_short_flag=run_short_flag,overwrite_flag=True)
# Do the TPT solve
tpt = TPT(nshort,lag_time_current,lag_time_seq,long_simfolder,short_simfolder,savefolder)
tpt.label_x_long(odw)
data = tpt.compile_data(odw)
function = function_obj.MSMBasis(basis_size,max_clust_per_level=100,min_clust_size=10)
tpt.compute_change_of_measure(odw,data,function)
tpt.compute_dam_moments_abba(odw,data,function,num_moments=4)
tpt.write_compare_generalized_rates(odw,data,suffix='0')
# End of solve 0
# Now resample according to residuals
nshort_new = int(0.5*nshort)
keys = list(odw.dam_dict.keys())
new_seed_weights = tpt.dam_moments[keys[0]]['res_xb']
_ = odw.generate_data_short(data.X[:,0],simfolder,tmax_short,nshort+nshort_new,dt_save,new_seed_weights,run_short_flag=True,overwrite_flag=False)
tpt = TPT(nshort,lag_time_current,lag_time_seq,long_simfolder,short_simfolder,savefolder)
tpt.label_x_long(odw)
data = tpt.compile_data(odw)
function = function_obj.MSMBasis(basis_size,max_clust_per_level=100,min_clust_size=10)
tpt.compute_change_of_measure(odw,data,function)
tpt.compute_dam_moments_abba(odw,data,function,num_moments=4)
tpt.write_compare_generalized_rates(odw,data,suffix='1')
# End of solve 1
sys.exit()
# ----------------------------
# Computations
if compute_tpt_flag:
    tpt.compute_change_of_measure(odw,data,function)
    tpt.compute_dam_moments_abba(odw,data,function,num_moments=4) 
    #tpt.compute_dam_moments_ab(odw,data,function,num_moments=4)
    #tpt.compute_comm_general(odw,data,function)
    #tpt.compute_mfpt_moments_ab(odw,data,function)
    #tpt.compute_conditional_mfpt_ab_moments(odw,data,function)
    tpt.write_compare_generalized_rates(odw,data)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
tpt.write_compare_generalized_rates(odw,data)
pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
# -------------------------------------------
# Displays
theta_1d_fun = lambda x: x[:,:1]
theta_1d_name = r"$x$"
theta_1d_units = 1.0
theta_2d_fun = lambda x: x[:,:2]
theta_2d_names = [r"$x$",r"$t$"]
theta_2d_units = np.ones(2)
tpt.display_dam_moments_abba(odw,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
pdf2png(1)
# ----------------------------
tpt.compute_compare_rate(odw,data)
# Displays
# Load long trajectory to compare to reality
t_long,x_long = odw.load_long_traj(long_simfolder)
x1_fun = lambda x: x[:,0]
tpt.plot_field_long(odw,data,data.X[:,0,0],r"$x_1$","x1",field_fun=x1_fun,units=1.0)
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
pdf2png(0)



