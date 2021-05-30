# This is where the double well potential is specified
import numpy as np
from os import mkdir
from os.path import join,exists
from numpy import save,load
from model_obj import Model
import helper

class DoubleWellModel(Model):
    def __init__(self,params): #state_dim=3,tau=0.25,kappa=0.0,sigma=1.0):
        dt_sim = params['dt_sim']
        self.q = {
            'tau': params['tau'],
            'kappa': params['kappa'],
            'sigma': params['sigma'],
            }
        state_dim = params['state_dim'],
        tpt_obs_dim = params['obs_dim'],
        noise_rank = params['state_dim'],
        tpt_obs_dim = params['obs_dim'],
        parallel_sim_limit = 50000
        nshort_per_file_limit = 100000
        super().__init__(state_dim,noise_rank,dt_sim,tpt_obs_dim,parallel_sim_limit,nshort_per_file_limit)
        x0_list = self.approximate_fixed_points()
        self.find_fixed_points(tmax=10)
        return
    def drift_fun(self,x):
        q = self.q
        Nx = len(x)
        b = np.zeros((Nx,self.state_dim))
        b[:,0] = 1/q['tau']*(x[:,0] - x[:,0]**3 + q['kappa'])
        for d in range(1,self.state_dim):
            b[:,d] = -1/q['tau']*x[:,d]
        return b
    def drift_jacobian_fun(self,x):
        # single x
        bx = np.zeros((self.state_dim,self.state_dim))
        bx[0,0] = 1/q['tau']*(1 - 3*x[0]**2)
        for d in range(1,self.state_dim):
            bx[d,d] = -1/q['tau']
        return bx
    def diffusion_fun(self,x):
        q = self.q
        Nx = len(x)
        wdot = np.random.randn(self.state_dim*Nx).reshape((self.state_dim,Nx))
        sig = q['sigma']*np.eye(self.state_dim)
        sw = (sig.dot(wdot)).T
        return sw
    def diffusion_mat(self,q):
        return q['sigma']*np.eye(self.state_dim)
    def tpt_observables(self,x):
        cvx = x
        if cvx.shape[1] != self.tpt_obs_dim:
            sys.exit("ERROR: tpt_observables output does not match expected dimension")
        return x
    def sampling_features(self,x):
        # x must be the output of tpt_observables
        return x
    def sampling_density(self,x):
        # Return the desired target density
        return np.ones(len(x))
    def create_tpt_damage_functions(self):
        # A dictionary of lambda functions of interest to be integrated along reactive trajectories (or just forward-reactive or backward-reactive).
        q = self.q
        self.dam_dict = {
                'one': {
                    'pay': lambda x: np.ones(len(x)),
                    'name_fwd': "\\tau^+", #r"$\tau^+$",
                    'name_bwd': "\\tau^-", #r"$\tau^-$",
                    'name_full': "\\tau", #r"$\tau^+-\tau^-$",
                    'abb_fwd': 't+',
                    'abb_bwd': 't-',
                    'abb_full': 'tfull',
                    'units': 1.0,
                    'unit_symbol': "",
                    'logscale': True,
                    },
                #'potential': {
                #    'pay': lambda x: 1/q['tau']*(-x[:,0]**2/2 + x[:,0]**4/4 - q['kappa']*x[:,0] + np.sum(x[:,1:]**2/2,1)),
                #    'name_fwd': "V+", #r"$\int_0^{\tau^+}V(X(r))dr$",
                #    'name_bwd': "V-", #r"$\int_{\tau^-}^0V(X(r))dr$",
                #    'name_full': "V", #r"$\int_{\tau^-}^{\tau^+}V(X(r))dr$",
                #    'abb_fwd': 'V+',
                #    'abb_bwd': 'V-',
                #    'abb_full': 'Vfull',
                #    'units': 1.0,
                #    'unit_symbol': '',
                #    'logscale': False,
                #    },
                }
        return
    def approximate_fixed_points(self):
        xst_approx = np.zeros((2,self.state_dim))
        xst_approx[0,0] = -1.0/10 + 0.01*(2*np.random.rand()-1)
        xst_approx[1,0] = 1.0/10 + 0.01*(2*np.random.rand()-1)
        return xst_approx
    def adist(self,cvx):
        radius_a = 0.5
        cva = self.tpt_observables(self.xst[:1])
        da = np.sqrt(np.sum((cvx - cva)**2, 1))
        return np.maximum(0, da-radius_a)
    def bdist(self,cvx):
        radius_b = 0.5
        cvb = self.tpt_observables(self.xst[1:])
        db = np.sqrt(np.sum((cvx - cvb)**2, 1))
        return np.maximum(0, db-radius_b)
    def set_param_folder(self):
        self.param_foldername = ("tau{}_kappa{}_sigma{}_statedim{}".format(self.q['tau'],self.q['kappa'],self.q['sigma'],self.state_dim)).replace('.','p')
        return
    def regression_features(self,cvx):
        return cvx
