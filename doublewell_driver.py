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
import helper
from data_obj import Data
from basis_obj import Basis
import function_obj 
from tpt_obj import TPT
from shutil import copyfile

# -----------------------
#    Make folders
datafolder = "/scratch/jf4241/SHORT_doublewell"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-03-11")
if not exists(dayfolder): mkdir(dayfolder)
expfolder = join(dayfolder,"1")
if not exists(expfolder): mkdir(expfolder)

asymb = r"$\mathbf{a}$"
bsymb = r"$\mathbf{b}$"

# ------------------------
# Decide what to do
least_action_flag = 1
run_long_flag =     1
run_short_flag =    1
compute_tpt_flag =  1
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
dw = DoubleWellModel(physical_params)
# 2. Run long and short trajectories
long_simfolder,t_long,x_long = dw.generate_data_long(simfolder,tmax_long,dt_save,run_long_flag=run_long_flag)
seed_weights = helper.reweight_data(x_long,dw.sampling_features,dw.sampling_density)

# --------------------------
# Run short trajectories, either with single- or multiprocessor
short_simfolder = dw.generate_data_short_multithreaded(x_long,simfolder,algo_params,seed_weights,run_short_flag=run_short_flag,overwrite_flag=False)

# ---------------------------------------------
# Initialize TPT
tpt = TPT(nshort,lag_time_current,lag_time_seq,physical_param_folder,long_simfolder,short_simfolder,savefolder)
#tpt.label_x_long(dw)
# Initialize data
data = tpt.compile_data(dw)
# Initialize function approximator as MSM basis
function = function_obj.MSMBasis(basis_size,max_clust_per_level=100,min_clust_size=10)
# ----------------------------
# Computations
if compute_tpt_flag:
    tpt.label_x_long(dw)
    tpt.compute_change_of_measure(dw,data,function)
    tpt.compute_dam_moments_abba(dw,data,function,num_moments=1) 
    tpt.write_compare_generalized_rates(dw,data)
    pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
tpt = pickle.load(open(join(savefolder,"tpt"),"rb"))
#tpt.write_compare_generalized_rates(dw,data)
#pickle.dump(tpt,open(join(savefolder,"tpt"),"wb"))
# -------------------------------------------
# Displays
theta_1d_fun = lambda x: x[:,:1]
theta_1d_name = r"$x_0$"
theta_1d_units = 1.0
theta_2d_fun = lambda x: x[:,:2]
theta_2d_names = [r"$x_0$",r"$x_1$"]
theta_2d_units = np.ones(2)
theta_2d_unit_symbols = ["",""]
tpt.display_change_of_measure_current(dw,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols)
tpt.display_change_of_measure(dw,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
tpt.display_dam_moments_abba_current(dw,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols)
tpt.display_dam_moments_abba(dw,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units)
pdf2png(1)
# ----------------------------
tpt.compute_compare_rate(dw,data)
# Displays
# Load long trajectory to compare to reality
t_long,x_long = dw.load_long_traj(long_simfolder)
x1_fun = lambda x: x[:,0]
tpt.plot_field_long(dw,data,data.X[:,0,0],r"$x_1$","x1",field_fun=x1_fun,units=1.0)
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


