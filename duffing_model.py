
# This is where the Duffing model is specified
import numpy as np
from numpy import load,save
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0
smallfont = {'family': 'serif', 'size': 10,}
font = {'family': 'serif', 'size': 18,}
ffont = {'family': 'serif', 'size': 25}
bigfont = {'family': 'serif', 'size': 40}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.sparse as sps
import time
import os
from os.path import join,exists
import sys
from model_obj import Model

class DuffingOscillator(Model):
    #def __init__(self,hB_d=38.5,du_per_day=1.0,dh_per_day=0.0,ref_alt=30.0,abdefdim=75):
    def __init__(self,physical_params,physical_param_string,xst=None):
        # dx = v*dt
        # dv = -(a*v + b*x + c*x**3)*dt + sigma*dW
        q = {}
        q['mu'] = physical_params['mu']
        q['kappa'] = physical_params['kappa']
        q['alpha'] = physical_params['alpha']
        q['sigma_x'] = physical_params['sigma_x']
        q['sigma_p'] = physical_params['sigma_p']
        q['dt_sim'] = physical_params['dt_sim']
        self.physical_param_string = physical_param_string
        self.state_dim = 2
        self.noise_rank = 2
        self.q = self.initialize_params(q)
        tpt_obs_dim = self.state_dim # We're dealing with full state here
        self.dt_sim = self.q['dt_sim']
        parallel_sim_limit = 10000
        nshort_per_file_limit = 100000
        super().__init__(self.state_dim,self.noise_rank,q['dt_sim'],tpt_obs_dim,parallel_sim_limit,nshort_per_file_limit)
        x0_list = self.approximate_fixed_points()
        if xst is None:
            self.find_fixed_points(tmax=600)
            print("self.xst = {}".format(self.xst))
        else:
            self.xst = xst
        self.tpt_obs_xst = self.tpt_observables(self.xst)
        return
    def initialize_params(self,q):
        q['sig_mat'] = np.zeros((self.state_dim,self.noise_rank))
        q['sig_mat'][0,0] = q['sigma_x']
        q['sig_mat'][1,1] = q['sigma_p']/q['mu'] # Momentum / mass
        return q
    def drift_fun(self,x):
        q = self.q
        drift = np.zeros((len(x),self.state_dim))
        drift[:,0] = x[:,1]
        drift[:,1] = 1/q['mu']*(-x[:,1] + q['alpha']*(x[:,0] - x[:,0]**3 + q['kappa']))
        return drift
    def drift_jacobian_fun(self,x):
        # x is just a single instance
        q = self.q
        J = np.zeros((self.state_dim,self.state_dim))
        J[0,1] = 1.0
        J[1,0] = 1/q['mu']*q['alpha']*(1 - 3*x[0]**2)
        J[1,1] = -1/q['mu']
        return J
    def diffusion_fun(self,x):
        wdot = np.random.randn(len(x)*self.noise_rank).reshape((self.noise_rank,len(x)))
        diff_term = (self.q['sig_mat'].dot(wdot)).T
        return diff_term
    def diffusion_mat(self,x):
        return self.q['sig_mat']
    def tpt_observables(self,x):
        return x # No reduction happening here
    def adist(self,cvx):
        cva = self.tpt_observables(self.xst[0])
        x_scale = 1.0
        v_scale = 0.25
        da = np.sqrt(((cvx[:,0]-cva[0])/x_scale)**2 + ((cvx[:,1]-cva[1])/v_scale)**2)
        #da = np.sqrt(np.sum((cvx - cva)**2, 1))
        radius_a = 0.4
        return np.maximum(0, da-radius_a)
    def bdist(self,cvx):
        cvb = self.tpt_observables(self.xst[1])
        x_scale = 1.0
        v_scale = 0.25
        db = np.sqrt(((cvx[:,0]-cvb[0])/x_scale)**2 + ((cvx[:,1]-cvb[1])/v_scale)**2)
        #db = np.sqrt(np.sum((cvx - cvb)**2, 1))
        radius_b = 0.4
        return np.maximum(0, db-radius_b)
    def set_param_folder(self):
        self.param_foldername = self.physical_param_string #("sig{}".format(self.q['sigma'])).replace(".","p")
        return
    def plot_least_action(self,physical_param_folder,fun_name="U"):
        funlib = self.observable_function_library()
        # Given the noise forcing, plot a picture of the least action pathway
        q = self.q
        n = q['Nz']-1
        sig = q['sig_mat']
        obs_xst = funlib[fun_name]["fun"](self.tpt_obs_xst)
        units = funlib[fun_name]["units"]
        unit_symbol = funlib[fun_name]["unit_symbol"]
        # A -> B
        fig,ax = plt.subplots(nrows=3,ncols=2,figsize=(18,18),sharex=True,constrained_layout=True)
        wmin = load(join(physical_param_folder,"wmin_dirn1.npy"))
        xmin = load(join(physical_param_folder,"xmin_dirn1.npy"))
        tmin = load(join(physical_param_folder,"tmin_dirn1.npy"))
        obs = funlib[fun_name]["fun"](self.tpt_observables(xmin))
        dU = (sig.dot(wmin.T)).T[:,2*n:3*n] # This part is specific to U
        z = q['z_d'][1:-1]/1000
        tz,zt = np.meshgrid(tmin,z,indexing='ij')
        ax[0,0].plot(tmin,units*obs[:,q['zi']],color='black')
        ax[0,0].plot(tmin[[0,-1]],units*obs_xst[0,q['zi']]*np.ones(2),color='skyblue')
        ax[0,0].plot(tmin[[0,-1]],units*obs_xst[1,q['zi']]*np.ones(2),color='red')
        im = ax[1,0].contourf(tz,zt,units*obs,cmap=plt.cm.coolwarm)
        im = ax[2,0].contourf(tz[:-1,:],zt[:-1,:],dU,cmap=plt.cm.coolwarm)
        # B -> A
        wmin = load(join(physical_param_folder,"wmin_dirn-1.npy"))
        xmin = load(join(physical_param_folder,"xmin_dirn-1.npy"))
        tmin = load(join(physical_param_folder,"tmin_dirn-1.npy"))
        obs = funlib[fun_name]["fun"](self.tpt_observables(xmin))
        dU = (sig.dot(wmin.T)).T[:,2*n:3*n]
        z = q['z_d'][1:-1]/1000
        tz,zt = np.meshgrid(tmin,z,indexing='ij')
        ax[0,1].plot(tmin,units*obs[:,q['zi']],color='black')
        ax[0,1].plot(tmin[[0,-1]],units*obs_xst[0,q['zi']]*np.ones(2),color='skyblue')
        ax[0,1].plot(tmin[[0,-1]],units*obs_xst[1,q['zi']]*np.ones(2),color='red')
        im = ax[1,1].contourf(tz,zt,units*obs,cmap=plt.cm.coolwarm)
        im = ax[2,1].contourf(tz[:-1,:],zt[:-1,:],dU,cmap=plt.cm.coolwarm)
        # Common legends
        #fig.suptitle("Least action paths",fontdict=bigfont)
        ax[0,0].set_title(r"$A\to B$ least action",fontdict=bigfont)
        ax[0,1].set_title(r"$B\to A$ least action",fontdict=bigfont)
        ax[1,0].set_title("%s"%(funlib[fun_name]["name"]),fontdict=bigfont)
        ax[1,1].set_title("%s"%(funlib[fun_name]["name"]),fontdict=bigfont)
        ax[2,0].set_title(r"$\delta U(t)$",fontdict=bigfont)
        ax[2,1].set_title(r"$\delta U(t)$",fontdict=bigfont)
        ax[0,0].set_ylabel("%s(%d km) (%s)"%(funlib[fun_name]["name"],self.ref_alt,funlib[fun_name]["unit_symbol"]),fontdict=bigfont)
        ax[1,0].set_ylabel(r"$z\,(\mathrm{km})$",fontdict=bigfont)
        ax[2,0].set_ylabel(r"$z\,(\mathrm{km})$",fontdict=bigfont)
        # Tick labels
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i,j].tick_params(axis='both',labelsize=30)
        # Save
        fig.savefig(join(physical_param_folder,"fw_plot_%s"%fun_name))
        plt.close(fig)
        return 
    def sampling_features(self,x,algo_params):
        # x must be the output of tpt_observables
        funlib = self.observable_function_library()
        Nx,xdim = x.shape
        names = algo_params['sampling_feature_names']
        samp_feat = np.zeros((Nx,len(names)))
        for i in range(len(names)):
            samp_feat[:,i] = funlib[names[i]]["fun"](x).flatten()
        #samp_feat[:,0] = funlib["vTintref"]["fun"](x).flatten()
        #samp_feat[:,1] = funlib["Uref"]["fun"](x).flatten()
        return samp_feat
    def sampling_density(self,x):
        return np.ones(len(x))
    def create_tpt_damage_functions(self):
        # A dictionary of lambda functions of interest to be integrated along reactive trajectories (or just forward-reactive or backward-reactive).
        q = self.q
        funlib = self.observable_function_library()
        self.corr_dict = {
                'v_g0': {
                    'pay': lambda x: 1.0*(x[:,1] > 0), #np.ones(len(x)),
                    'name': "v\\geq0",
                    },
                'v_l0': {
                    'pay': lambda x: 1.0*(x[:,1] < 0), #np.ones(len(x)),
                    'name': "v\\leq0",
                    },
                }
        self.dam_dict = {
                'one': {
                    'pay': lambda x: np.ones(len(x)),
                    'name': 'Time',
                    'name_fwd': "\\tau^+", #r"$\tau^+$",
                    'name_bwd': "\\tau^-", #r"$\tau^-$",
                    'name_full': "\\tau^+-\\tau^-", #r"$\tau^+-\tau^-$",
                    'abb_fwd': 't+',
                    'abb_bwd': 't-',
                    'abb_full': 'tfull',
                    'units': 1.0,
                    'unit_symbol': "\\mathrm{days}",
                    'unit_symbol_t': "\\mathrm{days}",
                    'logscale': True,
                    'pay_symbol': "1",
                    },
                }
        return
    def approximate_fixed_points(self):
        x = np.array([[-1.0,0.0],[1.0,0.0]],dtype=float)
        return x
    def regression_feature_names(self):
        funlib = self.observable_function_library()
        theta_names = [r"Re$\{\Psi\}$",r"Im$\{\Psi\}$",r"$U$",funlib["vTint"]["name"],funlib["vq"]["name"],funlib["dqdy"]["name"]][0:]
        return theta_names
    def regression_features(self,x):
        funlib = self.observable_function_library()
        n = self.q['Nz']-1
        Nx = len(x)
        lass = np.zeros((Nx,6*n))
        lass[:,:3*n] = x
        lass[:,3*n:4*n] = funlib["vTint"]["fun"](x)
        lass[:,4*n:5*n] = funlib["vq"]["fun"](x)
        lass[:,5*n:6*n] = funlib["dqdy"]["fun"](x)
        return lass[:,0:]
    def observable_function_library(self):
        q = self.q
        # Create a library of observable functions
        zlevel = lambda z: np.argmin(np.abs(q['z_d'][1:-1]/1000-z))
        funs = {
                "x":
                {"fun": lambda X: X[:,0],
                 "name":r"$x$",
                 "units": 1.0,
                 "unit_symbol": r""
                 },
                "v":
                {"fun": lambda X: X[:,1],
                 "name":r"$v$",
                 "units": 1.0,
                 "unit_symbol": r""
                 },
            }
        return funs

