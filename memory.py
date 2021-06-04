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
import hm_params_memory
import helper
from data_obj import Data
from tpt_obj import TPT

# ----------- Make folders ----------
datafolder = "/scratch/jf4241/SHORT_holtonmass"
if not exists(datafolder): mkdir(datafolder)
simfolder = join(datafolder,"runs")
if not exists(simfolder): mkdir(simfolder)
resultfolder = join(datafolder,"results_memory")
if not exists(resultfolder): mkdir(resultfolder)
dayfolder = join(resultfolder,"2021-06-03")
if not exists(dayfolder): mkdir(dayfolder)
expfolder = join(dayfolder,"0")
if not exists(expfolder): mkdir(expfolder)
# -----------------------------------

asymb = r"$\mathbf{a}$"
bsymb = r"$\mathbf{b}$"

# ---------- Decide what to do ----------
find_fixed_points_flag = 0
run_long_flag =          0
least_action_flag =      0
label_long_flag =        1
# ---------------------------------------

# ---------- Set parameters --------------------------
algo_params,algo_param_string = hm_params_memory.get_algo_params()
physical_params,physical_param_string = hm_params_memory.get_physical_params()
# Set savefolder accordingly
physical_param_folder = join(expfolder,physical_param_string)
if not exists(physical_param_folder): mkdir(physical_param_folder)
savefolder = join(physical_param_folder,algo_param_string)
if not exists(savefolder): mkdir(savefolder)
copyfile(join(codefolder,"hm_params_memory.py"),join(savefolder,"hm_params_memory.py"))
# -----------------------------------------------------
np.random.seed(0)
# ---------- 1. Initialize the model ----------
xst = None
if not find_fixed_points_flag:
    xst = np.load(join(savefolder,"xst.npy"))
model = HoltonMassModel(physical_params,xst=xst)
np.save(join(savefolder,"xst"),model.xst)
q = model.q # Dictionary of model parameters
# -------------------------------------------

# ---------- 2. Find the least action pathway ----------
if least_action_flag:
    model_lap = HoltonMassModel(physical_params)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=-1,maxiter=10)
    model_lap.minimize_action(100.0,physical_param_folder,dirn=1,maxiter=10)
    # Plot least action
    model.plot_least_action(physical_param_folder,"U")
    model.plot_least_action(physical_param_folder,"mag")
    model.plot_least_action(physical_param_folder,"vTint")
# -------------------------------------------------------

# --------------- Long simulation --------------
print("About to run long trajectory")
long_simfolder,t_long,x_long = model.generate_data_long(simfolder,algo_params,run_long_flag=run_long_flag)
seed_weights = helper.reweight_data(x_long,model.sampling_features,algo_params,model.sampling_density)
# ----------------------------------------------

# --------------- Collect the transition pathways -----------------
short_simfolder = None
tpt = TPT(algo_params,physical_param_folder,long_simfolder,short_simfolder,savefolder)
if label_long_flag:
    ab_starts,ab_ends,ba_starts,ba_ends,dam_emp = tpt.label_x_long(model)
    np.save(join(savefolder,"ab_starts"),ab_starts)
    np.save(join(savefolder,"ab_ends"),ab_ends)
    np.save(join(savefolder,"ba_starts"),ba_starts)
    np.save(join(savefolder,"ba_ends"),ba_ends)
    pickle.dump(dam_emp,open(join(savefolder,"dam_emp"),"wb"))
ab_starts = np.load(join(savefolder,"ab_starts.npy"))
ab_ends = np.load(join(savefolder,"ab_ends.npy"))
ba_starts = np.load(join(savefolder,"ba_starts.npy"))
ba_ends = np.load(join(savefolder,"ba_ends.npy"))
dam_emp = pickle.load(open(join(savefolder,"dam_emp"),"rb"))
t_long,x_long = model.load_long_traj(long_simfolder)
# -----------------------------------------------------------------
print("len(ab_starts) = {}".format(len(ab_starts)))
print("ab_starts[0:2] = {}, ba_starts[0:2] = {}".format(ab_starts[0:2],ba_starts[0:2]))
abfirst = (ab_starts[0] < ba_starts[0])
if abfirst:
    ab_offset = 0
    ba_offset = 0.5
else:
    ba_offset = 0
    ab_offset = 0.5
nab = len(ab_starts)
nba = len(ba_starts)

# Basic stuff: plot sequential durations
ab_durations = t_long[ab_ends] - t_long[ab_starts]
ba_durations = t_long[ba_ends] - t_long[ba_starts]
trans2plot = min(100,min(nab,nba))
fig,ax = plt.subplots()
hab, = ax.plot(ab_offset+np.arange(trans2plot),ab_durations[:trans2plot],color='darkorange',label=r"$A\to B$")
hba, = ax.plot(ba_offset+np.arange(trans2plot),ba_durations[:trans2plot],color='mediumspringgreen',label=r"$B\to A$")
ax.legend(handles=[hab,hba])
ax.set_title("Transition durations")
ax.set_xlabel("Transition number")
ax.set_ylabel("Duration (days)")
fig.savefig(join(savefolder,"durations"))
plt.close(fig)

keys = list(dam_emp.keys())
for k in range(len(keys)):
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))
    hab, = ax[0].plot(ab_offset+np.arange(trans2plot),dam_emp[keys[k]]['ab'][:trans2plot],color='darkorange',label=keys[k])
    hba, = ax[0].plot(ba_offset+np.arange(trans2plot),dam_emp[keys[k]]['ba'][:trans2plot],color='mediumspringgreen',label=keys[k])
    ax[0].legend(handles=[hab,hba])
    ax[0].set_title("Transitions %s"%keys[k])
    ax[1].scatter(dam_emp[keys[k]]['ab'][:-1],dam_emp[keys[k]]['ab'][1:],color='darkorange')
    ax[1].set_xlabel(r"%s$_k$"%(keys[k]))
    ax[1].set_ylabel(r"%s$_{k+1}$"%(keys[k]))
    ax[1].set_title(r"$A\to B$ memory")
    ax[2].scatter(dam_emp[keys[k]]['ba'][:-1],dam_emp[keys[k]]['ba'][1:],color='mediumspringgreen')
    ax[2].set_xlabel(r"%s$_k$"%(keys[k]))
    ax[2].set_ylabel(r"%s$_{k+1}$"%(keys[k]))
    ax[2].set_title(r"$B\to A$ memory")
    fig.savefig(join(savefolder,"dam{}".format(k)))
    plt.close(fig)


