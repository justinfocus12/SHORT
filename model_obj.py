# This file is where the model is defined
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from numpy import save,load
import multiprocessing
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import join,exists
codefolder = "/home/jf4241/dgaf2"
os.chdir(codefolder)
import helper
import sys
from abc import ABC,abstractmethod


class Model(ABC):
    def __init__(self,state_dim,noise_rank,dt_sim,tpt_obs_dim,parallel_sim_limit,nshort_per_file_limit):
        self.state_dim = state_dim
        self.noise_rank = noise_rank
        self.dt_sim = dt_sim
        self.tpt_obs_dim = tpt_obs_dim
        self.parallel_sim_limit = parallel_sim_limit
        self.nshort_per_file_limit = nshort_per_file_limit
        self.create_tpt_damage_functions()
        super().__init__()
        return
    @abstractmethod
    def drift_fun(self,x):
        pass
    @abstractmethod
    def drift_jacobian_fun(self,x):
        pass
    @abstractmethod
    def diffusion_fun(self,x):
        # The random perturbation that gets multiplied by dt
        pass
    @abstractmethod
    def diffusion_mat(self,x):
        # This applies specifically to Ito diffusions. It's an assumption on the structure of diffusion_fun
        pass
    @abstractmethod
    def tpt_observables(self,x):
        # This reduces the data to some CV space on which the DGA algorithm can act
        # This might change repeatedly even when analyzing the same dataset
        pass
    @abstractmethod
    def sampling_features(self,x):
        # Define the features on which the resampling is based
        pass
    @abstractmethod
    def sampling_density(self,x):
        pass
    @abstractmethod
    def create_tpt_damage_functions(self):
        # Make a dictionary of damage functions for use in TPT
        pass
    @abstractmethod
    def adist(self,cvx):
        pass
    @abstractmethod
    def bdist(self,cvx):
        pass
    def integrate_euler_maruyama_many(self,x0,t_save,stochastic_flag=True,max_chunk_size=5000,print_interval=1):
        # In the case of a TON of initial conditions, split it up into chunks
        num_traj = len(x0)
        x = np.zeros((len(t_save),num_traj,self.state_dim))
        n = 0
        while n < num_traj:
            idxrange = np.arange(n, min(n+max_chunk_size,num_traj))
            print("Starting trajectories {}-{} out of {}".format(idxrange[0],idxrange[-1],num_traj))
            #print("x0[idxrange].shape = {}".format(x0[idxrange].shape))
            new_traj = self.integrate_euler_maruyama(x0[idxrange],t_save,stochastic_flag=stochastic_flag,print_interval=print_interval)
            x[:,idxrange,:] = new_traj
            n += len(idxrange)
        return x
    def integrate_euler_maruyama(self,x0,t_save,stochastic_flag=True,print_interval=None):
        # Integrate with Euler-Maruyama. Computational timestep of dt_sim,
        # and save times of t_save
        # x0 is an array of initial conditions
        num_traj,xdim = x0.shape
        Nt_save = len(t_save)
        Nt_sim = int(np.ceil((t_save[-1]-t_save[0])/self.dt_sim)) + 1
        self.dt_sim = (t_save[-1] - t_save[0])/(Nt_sim - 1)
        x = np.zeros((Nt_save,num_traj,self.state_dim))
        x[0] = x0
        xold = x0
        ti = 1 # Next save index
        told = t_save[0]
        tnew = told
        while tnew < t_save[-1]:
            xnew = xold + self.dt_sim*self.drift_fun(xold)
            if stochastic_flag:
                xnew += np.sqrt(self.dt_sim)*self.diffusion_fun(xold)
            #print("xnew - xold = {}".format(xnew - xold))
            tnew = told + self.dt_sim
            #print("tnew = {}".format(tnew))
            while (ti < Nt_save) and (t_save[ti] <= tnew):
                frac = (t_save[ti] - told)/(tnew - told)
                x[ti] = xold*(1 - frac) + xnew*frac 
                ti += 1
            if print_interval is not None:
                if (tnew // print_interval) > (told // print_interval):
                    print("Time {:3.3f} out of {:.3f} ({:3.3f}% done)".format(tnew,t_save[-1],tnew/t_save[-1]*100))
            told = tnew
            xold = xnew
        return x
    @abstractmethod
    def approximate_fixed_points(self):
        pass
    def find_fixed_points(self,tmax=100):
        # From a list of two initial conditions, find the fixed points
        # Should generalize to find periodic orbits
        x0_list = self.approximate_fixed_points()
        t_save = np.linspace(0,tmax,10000)
        x = self.integrate_euler_maruyama(x0_list,t_save,stochastic_flag=False)
        if np.max(np.abs(x[-1]-x[-2])) > 1e-6 or np.max(np.abs(self.drift_fun(x[-1]))) > 1e-6:
            print("WARNING! Not converged to fixed points")
            print("Gradient magnitudes: {}".format(np.sqrt(np.sum((self.drift_fun(x[-1]))**2, 1))))
        self.xst = x[-1]
        self.xst_cv = self.tpt_observables(self.xst)
        return 
    @abstractmethod
    def set_param_folder(self):
        # Set a descriptive name including all the physical parameters
        pass
    def run_long_traj(self,long_simfolder,tmax_long,dt_save,print_interval=500):
        # Save a long ergodic trajectory into a specifically named subfolder in savedir
        x0 = np.zeros((1,self.state_dim))
        Nt = int(tmax_long/dt_save) + 1
        t_long = np.linspace(0,tmax_long,Nt)
        x_long = self.integrate_euler_maruyama(x0,t_long,print_interval=print_interval).reshape((Nt,self.state_dim))
        save(join(long_simfolder,"x_long"),x_long)
        save(join(long_simfolder,"t_long"),t_long)
        return
    def load_long_traj(self,long_simfolder):
        # This is a gatekeeper function for TPT: only give the TPT observables.
        x_long = load(join(long_simfolder,"x_long.npy"))
        cvx_long = self.tpt_observables(x_long)
        t_long = load(join(long_simfolder,"t_long.npy"))
        return t_long,cvx_long
    def load_short_traj(self,short_simfolder,num_traj):
        print("in load_short_traj: num_traj={}".format(num_traj))
        # The directory had better exist and have enough trajectories
        if not exists(short_simfolder):
            sys.exit("DOH! short_simfolder does not exist to load from")
        t_short = load(join(short_simfolder,"t_short.npy"))
        Nt = len(t_short)
        cvx_short = np.zeros((Nt,0,self.tpt_obs_dim))
        #sfli = np.zeros(0, dtype=int)
        num_loaded = 0
        i = 0
        while num_loaded < num_traj:
            x_short = load(join(short_simfolder,"x_short_{}.npy".format(i)))
            Nx_new = min(x_short.shape[1],num_traj-num_loaded)
            cvx_short_new = self.tpt_observables(x_short[:,:Nx_new,:].reshape((Nt*Nx_new,self.state_dim))).reshape((Nt,Nx_new,self.tpt_obs_dim))
            cvx_short = np.concatenate((cvx_short,cvx_short_new[:,:Nx_new,:]),axis=1)
            #sfli = np.concatenate((sfli,load(join(short_simfolder,"short_from_long_idx_{}.npy".format(i)))[:Nx_new]))
            num_loaded += Nx_new
            print("num_loaded = {}".format(num_loaded))
            i += 1
        print("cvx_short.shape = {}".format(cvx_short.shape))
        return t_short,cvx_short  #sfli
    def run_short_traj(self,tmax_short,dt_save,x_seed,short_suffix,seed_weights,nshort,x_savefile=None,t_savefile=None,save_x=False,save_t=False):
        np.random.seed(short_suffix)
        name = multiprocessing.current_process().name
        print("Process {} starting with seed {}".format(name,short_suffix))
        print("x_seed: min={}, max={}, mean={}".format(np.min(x_seed[:,0]),np.max(x_seed[:,0]),np.mean(x_seed[:,0])))
        #sample_weights = helper.reweight_data(x_seed,theta_fun,theta_pdf)
        idx = np.random.choice(np.arange(len(x_seed)),nshort,p=seed_weights/np.sum(seed_weights),replace=True)
        print("idx: min={}, max={}, mean={}".format(np.min(idx),np.max(idx),np.mean(idx)))
        xstart = x_seed[idx]
        print("xstart[0]: min={}, max={}, mean={}".format(np.min(x_seed[idx,0]),np.max(x_seed[idx,0]),np.mean(x_seed[idx,0])))
        Nt = int(tmax_short/dt_save) + 1
        t_short = np.linspace(0,tmax_short,Nt)
        x_short = self.integrate_euler_maruyama_many(xstart,t_short,max_chunk_size=self.parallel_sim_limit,print_interval=5.0)
        if save_x:
            save(x_savefile,x_short)
        if save_t:
            save(t_savefile,t_short)
        print("Process {} finishing with seed {}".format(name,short_suffix))
        return #x_short,t_short #,idx
    def generate_data_long(self,simfolder,algo_params,run_long_flag=True):
        self.set_param_folder()
        if not exists(join(simfolder,self.param_foldername)): mkdir(join(simfolder,self.param_foldername))
        # -----------------------------------
        # Long simulation
        tmax_long = algo_params['tmax_long']
        dt_save = algo_params['dt_save']
        long_simfolder = (join(simfolder,self.param_foldername,"long_t{}".format(tmax_long))).replace(".","p")
        if run_long_flag:
            if not exists(long_simfolder): mkdir(long_simfolder)
            self.run_long_traj(long_simfolder,tmax_long,dt_save)
        else:
            if not (exists(join(long_simfolder,"x_long.npy")) and exists(join(long_simfolder,"t_long.npy"))):
                sys.exit("DOH! You don't want to run long, but the files don't exist.")
        x_long = load(join(long_simfolder,"x_long.npy"))
        t_long = load(join(long_simfolder,"t_long.npy"))
        return long_simfolder,t_long,x_long
    def generate_data_short_multithreaded(self,x_seed,simfolder,algo_params,seed_weights,run_short_flag=True,overwrite_flag=False):
        tmax_short = algo_params['tmax_short']
        nshort = algo_params['nshort']
        dt_save = algo_params['dt_save']
        # Given some base folder, optionally generate data. Throw exceptions if a flag is off and the data is not there
        # Top off the existing data if needed. 
        # -----------------------------------
        # Short simulation
        short_simfolder = (join(simfolder,self.param_foldername,"short_t{}".format(tmax_short))).replace(".","p")
        if (not run_short_flag) and exists(short_simfolder): return short_simfolder
        if run_short_flag and not exists(short_simfolder): mkdir(short_simfolder)
        if overwrite_flag:
            flist = os.listdir(short_simfolder)
            for f in flist:
                os.remove(join(short_simfolder,f))
        # Check what already exists
        x_short_filelist_existing = [f for f in os.listdir(short_simfolder) if f.startswith('x_short')]
        nshort_existing = 0
        for i in range(len(x_short_filelist_existing)):
            x_short = load(join(short_simfolder,x_short_filelist_existing[i]))
            nshort_existing += x_short.shape[1]
        if nshort_existing >= nshort:
            print("Returning early because nshort_existing >= nshort")
            return short_simfolder
        # If there is more to be done, generate another list
        num_files = int(np.ceil((nshort-nshort_existing)/self.nshort_per_file_limit))
        x_short_filelist_new = []
        idx0 = 0
        print("x_short_filelist_existing = {}".format(x_short_filelist_existing))
        short_suffix_list = np.arange(num_files) + len(x_short_filelist_existing)
        num_traj_list = np.zeros(num_files, dtype=int)
        for i in range(num_files):
            idx1 = min(nshort,idx0+self.nshort_per_file_limit)
            x_short_filelist_new += ["x_short_{}".format(short_suffix_list[i])]
            num_traj_list[i] = idx1-idx0
            idx0 = idx1
        if run_short_flag:
            jobs = []
            for i in range(num_files): # Parallelize this loop
                x_savefile = join(short_simfolder,x_short_filelist_new[i])
                t_savefile = join(short_simfolder,"t_short")
                proc = multiprocessing.Process(name="Short batch {}".format(i),target=self.run_short_traj,args=(tmax_short,dt_save,x_seed,short_suffix_list[i],seed_weights,num_traj_list[i],x_savefile,t_savefile,True,(i==0 and nshort_existing==0)))
                proc.daemon = (i==0)
                jobs.append(proc)
                proc.start()
            for i in range(len(jobs)): 
                jobs[i].join()
        print("Done with generating short data")
        return short_simfolder
    def generate_data_short(self,x_seed,simfolder,tmax_short,nshort,dt_save,seed_weights,run_short_flag=True,overwrite_flag=False):
        # Given some base folder, optionally generate data. Throw exceptions if a flag is off and the data is not there
        # new paradigm: top off the existing data if needed. 
        # -----------------------------------
        # Short simulation
        short_simfolder = (join(simfolder,self.param_foldername,"short_t{}".format(tmax_short))).replace(".","p")
        if run_short_flag and not exists(short_simfolder): mkdir(short_simfolder)
        print("does short_simfolder exist now? {}".format(exists(short_simfolder)))
        print("listdir(short_simfolder) = {}".format(os.listdir(short_simfolder)))
        if overwrite_flag:
            flist = os.listdir(short_simfolder)
            for f in flist:
                os.remove(join(short_simfolder,f))
        # Check what already exists
        x_short_filelist_existing = [f for f in os.listdir(short_simfolder) if f.startswith('x_short')]
        nshort_existing = 0
        for i in range(len(x_short_filelist_existing)):
            x_short = load(join(short_simfolder,x_short_filelist_existing[i]))
            nshort_existing += x_short.shape[1]
        if nshort_existing >= nshort:
            return short_simfolder
        # If there is more to be done, generate another list
        num_files = int(np.ceil((nshort-nshort_existing)/self.nshort_per_file_limit))
        x_short_filelist_new = []
        idx0 = 0
        print("x_short_filelist_existing = {}".format(x_short_filelist_existing))
        short_suffix_list = np.arange(num_files) + len(x_short_filelist_existing)
        for i in range(num_files):
            idx1 = min(nshort,idx0+self.nshort_per_file_limit)
            x_short_filelist_new += ["x_short_{}".format(short_suffix_list[i])]
            if run_short_flag:
                #x_short,t_short = self.run_short_traj(tmax_short,dt_save,x_seed,short_suffix_list[i],seed_weights,idx1-idx0)
                x_savefile = join(short_simfolder,x_short_filelist_new[i])
                t_savefile = join(short_simfolder,"t_short")
                self.run_short_traj(tmax_short,dt_save,x_seed,short_suffix_list[i],seed_weights,idx1-idx0,x_savefile,t_savefile,True,(i==0 and nshort_existing==0))
                #save(join(short_simfolder,x_short_filelist_new[i]),x_short) 
                #if i==0: save(join(short_simfolder,"t_short".format(short_suffix_list[i])),t_short)
                #save(join(short_simfolder,"short_from_long_idx_{}".format(short_suffix_list[i])),short_from_long_idx)
            #else:
            #    if not exists(join(short_simfolder,x_short_filename_list[i]+".npy")):
            #        sys.exit("DOH! You don't want to run short, but file {} doesn't exist".format(i))
            idx0 = idx1
        return short_simfolder
    def full_action(self,x_init,w,end_penalty):
        # Differentiate the end cost with respect to all the noise perturbations
        K = len(w) + 1 # Number of time steps
        x = self.noise2path(w,x_init)
        sqrtdt = np.sqrt(self.dt_sim)
        dPhi_dw = np.zeros((K-1,self.noise_rank))
        Phi,dPhi_dx = end_penalty(x[-1])
        sig_mat = self.diffusion_mat(self.xst[:1])
        for i in range(1,K):
            dPhi_dw[-i] = (sig_mat.T.dot(dPhi_dx)).T*sqrtdt 
            dPhi_dx = dPhi_dx.dot(np.eye(self.state_dim) + self.drift_jacobian_fun(x[-i-1])*self.dt_sim)
        path_act = 1/(2*K)*np.sum(w**2)
        path_act_der = 1/K*w
        return path_act,path_act_der,Phi,dPhi_dw
    def noise2path(self,w,x_init):
        K = len(w) + 1 # Number of time steps
        x = np.zeros((K,self.state_dim))
        x[0] = x_init
        sqrtdt = np.sqrt(self.dt_sim)
        print("self.xst[:1] = {}".format(self.xst[:1]))
        sig_mat = self.diffusion_mat(self.xst[:1])
        for i in range(K-1):
            x[i+1] = x[i] + self.drift_fun(x[i:i+1]).flatten()*self.dt_sim + sig_mat.dot(w[i])*sqrtdt
        return x
    def minimize_action(self,time_horizon,physical_param_folder,dirn=1,end_weight=1.0,maxiter=10):
        print("About to minimize action")
        # Find the least-action pathway between x_init and x_fin, of length time_horizon
        if dirn == 1:
            x_init,x_fin = self.xst
        else:
            x_fin,x_init = self.xst
        K = int(time_horizon/self.dt_sim)
        def end_penalty(x):
            Phi = np.sum((x-x_fin)**2)/2
            dPhi_dx = x-x_fin
            return Phi,dPhi_dx
        w0 = np.zeros((K-1)*self.noise_rank).reshape((K-1,self.noise_rank))
        # Optimize with the BFGS
        def func(wflat):
            wravel = wflat.reshape((K-1,self.noise_rank))
            path_act,path_act_der,Phi,dPhi_dw = self.full_action(x_init,wravel,end_penalty)
            value = path_act + end_weight*Phi
            jac = (path_act_der + end_weight*dPhi_dw).flatten()
            return value,jac
        print("About to L-BFGS")
        wmin,funcmin,optinfo = fmin_l_bfgs_b(func,w0.flatten(),maxiter=maxiter,iprint=1)
        print("Done with L-BFGS")
        wmin = wmin.reshape((K-1,self.noise_rank))
        np.save(join(physical_param_folder,"wmin_dirn{}".format(dirn)),wmin)
        x = self.noise2path(wmin,x_init)
        t = self.dt_sim*np.arange(K)
        np.save(join(physical_param_folder,"xmin_dirn{}".format(dirn)),x)
        np.save(join(physical_param_folder,"tmin_dirn{}".format(dirn)),t)
        # Maybe make some plots
        return 
    def load_least_action_path(self,physical_param_folder,dirn=1):
        # Convert to TPT observables
        xmin = self.tpt_observables(load(join(physical_param_folder,"xmin_dirn{}.npy".format(dirn))))
        tmin = load(join(physical_param_folder,"tmin_dirn{}.npy".format(dirn)))
        return xmin,tmin
        
