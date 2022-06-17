# Aggregate together statistics from different independent DGA runs. Also compare with ES results for a unified UQ pipeline.
import numpy as np
import time
from numpy import save,load
import pandas as pd
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 40
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
matplotlib.rcParams['legend.fontsize'] = 15
font = {'family': 'monospace', 'size': 20,}
bigfont = {'family': 'monospace', 'size': 40}
giantfont = {'family': 'monospace', 'size': 80}
ggiantfont = {'family': 'monospace', 'size': 120}
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
simfolder = join(datafolder,"runs")
resultfolder = join(datafolder,"results")
dayfolder = join(resultfolder,"2022-05-23")
expfolder = join(dayfolder,"0")
# -----------------------------------
algo_params,algo_param_string = hm_params.get_algo_params()
physical_params,physical_param_string = hm_params.get_physical_params()
# Set savefolder accordingly
physical_param_folder = join(expfolder,physical_param_string)

def compile_generalized_rates_dga(model,tpt_file_list,hm_params,algo_params,savefolder):
    # For each TPT output file, load it in and read the generalized rates
    Ntpt = len(tpt_file_list)
    keys = list(model.dam_dict.keys())
    Nkeys = len(keys)
    Nmom = algo_params["num_moments"]
    rates_dga = dict()
    for k in range(Nkeys):
        rates_dga[keys[k]] = {"ab": np.zeros((Ntpt,Nmom+1)), "ba": np.zeros((Ntpt,Nmom+1))}
    for i in range(Ntpt):
        tpt = pickle.load(open(tpt_file_list[i],"rb"))
        for k in range(Nkeys):
            rates_dga[keys[k]]["ab"][i,:] = tpt.dam_moments[keys[k]]['rate_ab']
            rates_dga[keys[k]]["ba"][i,:] = tpt.dam_moments[keys[k]]['rate_ba']
        if i == 0:
            # Extract ES info
            dam_dns = tpt.dam_emp
            long_from_label = tpt.long_from_label
            long_to_label = tpt.long_to_label
            t_long,x_long = model.load_long_traj(tpt.long_simfolder)
            del x_long
        del tpt
    ab_reactive_flag = 1*(long_from_label==-1)*(long_to_label==1)
    ba_reactive_flag = 1*(long_from_label==1)*(long_to_label==-1)
    num_rxn = np.sum(np.diff(ab_reactive_flag)==1)
    num_rxn = min(num_rxn,np.sum(np.diff(ba_reactive_flag)==1))
    print("num_rxn = {}".format(num_rxn))
    ab_starts = np.where(np.diff(ab_reactive_flag)==1)[0] + 1
    ab_ends = np.where(np.diff(ab_reactive_flag)==-1)[0] + 1
    ba_starts = np.where(np.diff(ba_reactive_flag)==1)[0] + 1
    ba_ends = np.where(np.diff(ba_reactive_flag)==-1)[0] + 1
    num_rxn = np.sum(np.diff(ab_reactive_flag)==1)
    num_rxn = min(num_rxn,np.sum(np.diff(ba_reactive_flag)==1))
    num_rxn -= 1 # For the periods
    Nt = len(t_long)
    dt = t_long[1] - t_long[0]
    print("num_rxn = {}".format(num_rxn))
    # ------------------- Generalized rates ---------------------
    # Look at time from one ab_start to the next, and treat that as the IID random variables. Do a bootstrap estimate
    Nboot = 500
    Nab = num_rxn #len(ab_starts)
    Nba = num_rxn #len(ba_starts)
    ret = {'ab': np.diff(ab_starts)*dt, 'ba': np.diff(ba_starts)*dt} # Time of each one. The fundamental unit. 
    genrate_dns = {}
    for key in keys:
        genrate_dns[key] = {}
        for dirn in ['ab','ba']:
            np.random.seed(1)
            genrate_dns[key][dirn] = {"mean": np.array([np.sum(dam_dns[key][dirn]**k)/np.sum(ret[dirn]) for k in range(Nmom+1)])}
            genrate_bootstrap = np.zeros((Nboot,Nmom+1))
            for i in range(Nboot):
                idx = np.random.choice(np.arange(Nab),size=Nab,replace=True)
                genrate_bootstrap[i] = np.array([np.sum(dam_dns[key][dirn][idx]**k)/np.sum(ret[dirn][idx]) for k in range(Nmom+1)])
            genrate_dns[key][dirn]["rmse"] = np.sqrt(np.mean((genrate_bootstrap - genrate_dns[key][dirn]["mean"])**2, axis=0))
    genrate_dga = {}
    for key in keys:
        genrate_dga[key] = {} 
        for dirn in ['ab','ba']:
            genrate_dga[key][dirn] = {"mean": np.mean(rates_dga[key][dirn], axis=0)}
            genrate_dga[key][dirn]["rmse"] = np.std(rates_dga[key][dirn], axis=0)
            genrate_dga[key][dirn]["min"] = np.min(rates_dga[key][dirn], axis=0)
            genrate_dga[key][dirn]["max"] = np.max(rates_dga[key][dirn], axis=0)
    dirns = ['ab','ba']
    # Initial figure for rates
    fig,ax = plt.subplots()
    names = [r"$A\to B$",r"$B\to A$"]
    index = ['ab','ba']
    yerr = np.zeros((2, 2, 2)) # (dns,dga) x (lo,hi) x (ab,ba) 
    yerr[0,:,:] = np.outer(np.ones(2), [2*genrate_dns[keys[0]][dirn]["rmse"][0] for dirn in index])
    yerr[1,0,:] = [genrate_dga[keys[0]][dirn]["mean"][0]-genrate_dga[keys[0]][dirn]["min"][0] for dirn in index]
    yerr[1,1,:] = [genrate_dga[keys[0]][dirn]["max"][0]-genrate_dga[keys[0]][dirn]["mean"][0] for dirn in index]
    print("yerr = \n{}".format(yerr))
    df = pd.DataFrame(index=index,data=dict({
        "Phase": names,
        "DGA": [genrate_dga[keys[0]][dirn]["mean"][0] for dirn in index],
        "ES": [genrate_dns[keys[0]][dirn]["mean"][0] for dirn in index],
        }))
    df.plot(kind="bar",x="Phase",y=['ES','DGA'],yerr=yerr,ax=ax,color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
    ax.set_title("Rate",fontdict=font)
    ax.set_ylabel(r"Rate [days$^{-1}$]",fontdict=font)
    ax.tick_params(axis='x',labelsize=15)
    ax.tick_params(axis='y',labelsize=15)
    ax.set_xlabel("")
    fig.savefig(join(savefolder,"rates_bar"),bbox_inches="tight",pad_inches=0.2)
    plt.close(fig)
    for key in keys:
        for i_mom in range(1,Nmom+1):
            fig,ax = plt.subplots()
            yerr = np.zeros((2, 2, 2)) # (dns,dga) x (lo,hi) x (ab,ba) 
            yerr[0,:,:] = np.outer(np.ones(2), [2*genrate_dns[key][dirn]["rmse"][i_mom] for dirn in index])
            yerr[1,0,:] = [genrate_dga[key][dirn]["mean"][i_mom]-genrate_dga[key][dirn]["min"][i_mom] for dirn in index]
            yerr[1,1,:] = [genrate_dga[key][dirn]["max"][i_mom]-genrate_dga[key][dirn]["mean"][i_mom] for dirn in index]
            df = pd.DataFrame(index=index,data=dict({
                "Phase": names,
                "DGA": [genrate_dga[key][dirn]["mean"][i_mom] for dirn in index],
                "ES": [genrate_dns[key][dirn]["mean"][i_mom] for dirn in index],
                }))
            df.plot(kind="bar",x="Phase",y=['ES','DGA'],yerr=yerr,ax=ax,color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
            ax.set_title("Generalized rate: {}, moment {}".format(model.dam_dict[key]['name'],i_mom))
            ax.set_ylabel(r"$(%s)^{%i}/\mathrm{time}$ $[(%s)^{%i}/\mathrm{day}]$"%(model.dam_dict[key]['name_full'],i_mom,model.dam_dict[key]['unit_symbol_t'],i_mom))
            ax.set_xlabel("")
            fig.savefig(join(savefolder,"genrate_{}_{}".format(key,i_mom)),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
        for i_dirn in range(len(dirns)):
            dirn = dirns[i_dirn]
            mean = genrate_dns[key][dirn]["mean"]
            rmse = genrate_dns[key][dirn]["rmse"]
            print("{} {}".format(key,dirn))
            print("\tgenrate_dga: \n\tmean={}\n\trmse={}".format(genrate_dga[key][dirn]['mean'],genrate_dga[key][dirn]['rmse']))
            print("\tgenrate_dns: \n\tmean={}\n\trmse={}".format(genrate_dns[key][dirn]['mean'],genrate_dns[key][dirn]['rmse']))
        fig.savefig(join(savefolder,"rate_line_{}".format(key)),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
    # ------------------- Time fractions ---------------------------------
    Nboot = 500
    Nab = num_rxn #len(ab_starts)
    Nba = num_rxn #len(ba_starts)
    ret = {'ab': np.diff(ab_starts)*dt, 'ba': np.diff(ba_starts)*dt} # Time of each one. The fundamental unit. 
    timefrac_dns = {}
    timefrac_dga = {}
    phase_list = ['ab','ba','aa','bb']
    phase_list_names = [r"$A\to B$",r"$B\to A$",r"$A\to A$",r"$B\to B$"]
    from_list = [-1,1,-1,1]
    to_list = [1,-1,-1,1]
    bwd_key = ['ax','bx','ax','bx']
    fwd_key = ['xb','xa','xa','xb']
    # ES
    for i_ph in range(len(phase_list)):
        phase = phase_list[i_ph]
        from_label = from_list[i_ph]
        to_label = to_list[i_ph]
        np.random.seed(1)
        timefrac_dns[phase] = {"mean": np.mean((long_from_label==from_label)*(long_to_label==to_label))}
        timefrac_bootstrap = np.zeros(Nboot)
        for i in range(Nboot):
            idx = np.random.choice(np.arange(Nab),size=Nab,replace=True)
            phase_time_total = 0.0
            time_total = 0.0
            for j in idx:
                ti0,ti1 = ab_starts[j],ab_starts[j+1]
                phase_time_total += np.sum((long_from_label[ti0:ti1]==from_label)*(long_to_label[ti0:ti1]==to_label))*dt
                time_total += (ti1-ti0)*dt
            timefrac_bootstrap[i] = phase_time_total/time_total
        timefrac_dns[phase]["rmse"] = np.sqrt(np.mean((timefrac_bootstrap - timefrac_dns[phase]["mean"])**2, axis=0))
    # DGA
    timefrac_dga = {}
    for phase in phase_list:
        timefrac_dga[phase] = np.zeros(Ntpt)
    for i in range(Ntpt):
        tpt = pickle.load(open(tpt_file_list[i],"rb"))
        for i_ph in range(len(phase_list)):
            phase = phase_list[i_ph]
            print("Phase {}".format(phase))
            comm_bwd = tpt.dam_moments['one'][bwd_key[i_ph]][0,:,0]
            comm_fwd = tpt.dam_moments['one'][fwd_key[i_ph]][0,:,0]
            timefrac_dga[phase][i] = np.sum(tpt.chom * comm_bwd * comm_fwd)
            print("comm_bwd: min={}, max={}. comm_fwd: min={}, max={}. chom: min={}, max={}, sum={}. time_frac_dga = {} ".format(comm_bwd.min(),comm_bwd.max(),comm_fwd.min(),comm_fwd.max(),tpt.chom.min(),tpt.chom.max(),tpt.chom.sum(),timefrac_dga[phase][i]))
    df = pd.DataFrame(index=phase_list,data=dict({
        "Phase": phase_list_names,
        "ES": [timefrac_dns[phase]["mean"] for phase in phase_list],
        "DGA": [timefrac_dga[phase].mean() for phase in phase_list],
        }))
    yerr = np.zeros((2,2,len(phase_list))) # (dns,dga) x (lo,hi) x (aa,ab,bb,ba)
    yerr[0,:,:] = np.outer(np.ones(2),[2*timefrac_dns[phase]["rmse"] for phase in phase_list])
    yerr[1,0,:] = [timefrac_dga[phase].mean()-timefrac_dga[phase].min() for phase in phase_list]
    yerr[1,1,:] = [timefrac_dga[phase].max()-timefrac_dga[phase].mean() for phase in phase_list]
    fig,ax = plt.subplots()
    df.plot(kind="bar",x="Phase",y=['ES','DGA'],yerr=yerr,ax=ax,color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
    #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0,subs=[0.8,1.0,1.2,1.4,1.6]))
    ax.set_title("Phase durations",fontdict=font)
    ax.set_ylabel(r"Time fraction",fontdict=font)
    ax.set_xlabel("")
    ax.tick_params(axis='x',labelsize=15)
    ax.tick_params(axis='y',labelsize=15)
    fig.savefig(join(savefolder,"lifecycle_bar"),bbox_inches="tight",pad_inches=0.2)
    ax.set_yscale('log')
    ticks = [0.025,0.05,0.1,0.2,0.4]
    ticklabels = [str(tick) for tick in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    fig.savefig(join(savefolder,"lifecycle_bar_log"),bbox_inches="tight",pad_inches=0.2)
    plt.close(fig)
    return

if __name__ == "__main__":
    run_model = False # Do this the first time running in a new dayfolder 
    savefolder = join(physical_param_folder,algo_param_string.replace("istart0","allstart"))
    if not exists(savefolder): mkdir(savefolder)
    # Make the list of savefolders
    tpt_file_list = []
    istart_list = [0,3,6]
    #istart_list = [0,1,2,3,5,6,7,8,9]
    for istart in istart_list: #[0,3,6]:
        tpt_file_list += [join(physical_param_folder,algo_param_string,"tpt").replace("istart0","istart%i"%(istart))]
    if run_model:
        model = HoltonMassModel(physical_params)
        np.save(join(savefolder,"xst"),model.xst)
    else:
        xst = np.load(join(savefolder,"xst.npy"))
        model = HoltonMassModel(physical_params,xst=xst)
    compile_generalized_rates_dga(model,tpt_file_list,hm_params,algo_params,savefolder)



