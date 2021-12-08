# Aggregate together statistics from different independent DGA runs. Also compare with DNS results for a unified UQ pipeline.
import numpy as np
import time
from numpy import save,load
import pandas as pd
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 25
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
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
simfolder = join(datafolder,"runs")
resultfolder = join(datafolder,"results")
dayfolder = join(resultfolder,"2021-12-07")
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
            # Extract DNS info
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
    print("num_rxn = {}".format(num_rxn))
    # ------------------- Summary statistics for DNS ---------------------
    Nt = len(t_long)
    dt = t_long[1] - t_long[0]
    # 1. ----- Time ------
    # Look at time from one ab_start to the next, and treat that as the IID random variables. Do a bootstrap estimate
    Nboot = 100
    Nab = num_rxn #len(ab_starts)
    Nba = num_rxn #len(ba_starts)
    ret = {'ab': np.diff(ab_starts)*dt, 'ba': np.diff(ba_starts)*dt} # Time of each one. The fundamental unit. 
    genrate_dns = {}
    for key in keys:
        genrate_dns[key] = {}
        for dirn in ['ab','ba']:
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
    dirns = ['ab','ba']
    # Initial figure for rates
    fig,ax = plt.subplots()
    index = ['ab','ba']
    names = [r"$A\to B$",r"$B\to A$"]
    df = pd.DataFrame(index=index,data=dict({
        "Phase": names,
        "DGA": [genrate_dga[keys[0]][dirn]["mean"][0] for dirn in index],
        "DNS": [genrate_dns[keys[0]][dirn]["mean"][0] for dirn in index],
        "DGA_unc": [4*genrate_dga[keys[0]][dirn]["rmse"][0] for dirn in index],
        "DNS_unc": [4*genrate_dns[keys[0]][dirn]["rmse"][0] for dirn in index],
        }))
    df.plot(kind="bar",x="Phase",y=['DNS','DGA'],yerr=df[['DNS_unc','DGA_unc']].to_numpy().T,ax=ax,color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
    ax.set_title("Rates")
    fig.savefig(join(savefolder,"rates_bar"))
    plt.close(fig)
    for key in keys:
        for i_mom in range(1,Nmom+1):
            fig,ax = plt.subplots()
            df = pd.DataFrame(index=index,data=dict({
                "Phase": names,
                "DGA": [genrate_dga[key][dirn]["mean"][i_mom] for dirn in index],
                "DNS": [genrate_dns[key][dirn]["mean"][i_mom] for dirn in index],
                "DGA_unc": [4*genrate_dga[key][dirn]["rmse"][i_mom] for dirn in index],
                "DNS_unc": [4*genrate_dns[key][dirn]["rmse"][i_mom] for dirn in index],
                }))
            df.plot(kind="bar",x="Phase",y=['DNS','DGA'],yerr=df[['DNS_unc','DGA_unc']].to_numpy().T,ax=ax,color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
            ax.set_title("Generalized rate: {}, moment {}".format(key,i_mom))
            fig.savefig(join(savefolder,"genrate_{}_{}".format(key,i_mom)))
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
    #ret_dns_bootstrap = np.zeros(Nboot)
    #for i in range(Nboot):
    #    ret_dns_resampled = np.random.choice(ret_dns, size=len(ret_dns), replace=True)
    #    ret_dns_bootstrap[i] = np.mean(ret_dns_resampled)
    #ret_dns_mean = np.mean(ret_dns)
    #ret_dns_rmse = np.sqrt(np.mean((ret_dns_bootstrap - ret_dns_mean)**2))
    ## Collect the return time from DGA
    #ret_dga_ab = 1.0 / rates_dga[keys[0]]['ab'][:,0]
    #ret_dga_ba = 1.0 / rates_dga[keys[0]]['ba'][:,0]
    #ret_dga_ab_mid = (ret_dga_ab.min() + ret_dga_ab.max())/2 
    #ret_dga_ba_mid = (ret_dga_ba.min() + ret_dga_ba.max())/2
    #ret_dga_ab_unc = (ret_dga_ab.max() - ret_dga_ab.min()) 
    #ret_dga_ba_unc = (ret_dga_ba.max() - ret_dga_ba.min())
    ## Do a bar plot. First DNS, then DGA AB, then DGA BA
    #fig,ax = plt.subplots()
    #index = ['dns','dgaab','dgaba']
    #names = ['DNS',r'DGA $(A\to B)$',r'DGA $(B->A)$']
    #df = pd.DataFrame(index=index,data=dict({
    #    "Method": names,
    #    "Return period": [ret_dns_mean,ret_dga_ab_mid,ret_dga_ba_mid],
    #    "ret_unc": [4*ret_dns_rmse,ret_dga_ab_unc,ret_dga_ba_unc],
    #    }))
    #print(df)
    #df.plot(kind="bar",x="Method",y=["Return period"],yerr=df[["ret_unc"]].to_numpy().T,ax=ax,color=['cyan'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
    #ax.set_title("Rates")
    #fig.savefig(join(savefolder,"rate_bar"),bbox_inches="tight",pad_inches=0.2)
    #plt.close(fig)
    return

if __name__ == "__main__":
    run_model = False
    savefolder = join(physical_param_folder,algo_param_string.replace("istart0","allstart"))
    if not exists(savefolder): mkdir(savefolder)
    # Make the list of savefolders
    tpt_file_list = []
    for istart in [0,3,6]:
        tpt_file_list += [join(physical_param_folder,algo_param_string,"tpt").replace("istart0","istart%i"%(istart))]
    if run_model:
        model = HoltonMassModel(physical_params)
        np.save(join(savefolder,"xst"),model.xst)
    else:
        xst = np.load(join(savefolder,"xst.npy"))
        model = HoltonMassModel(physical_params,xst=xst)
    compile_generalized_rates_dga(model,tpt_file_list,hm_params,algo_params,savefolder)



