# This is where the oscillating double well potential is specified
import numpy as np
from os import mkdir
from scipy.interpolate import interp1d
from os.path import join,exists
from numpy import save,load
from model_obj import Model
import helper

class OscillatingDoubleWellModel(Model):
    def __init__(self,state_dim=2,tau=0.25,kappa=0.0,lam=0.5,sigma=1.0):
        dt_sim = 0.001
        self.q = {
            'tau': tau,
            'kappa': kappa,
            'sigma': sigma,
            'lam': lam,  # strngth of oscillation
            }
        noise_rank = state_dim
        tpt_obs_dim = 2
        parallel_sim_limit = 50000
        nshort_per_file_limit = 100000
        super().__init__(state_dim,noise_rank,dt_sim,tpt_obs_dim,parallel_sim_limit,nshort_per_file_limit)
        # Find the orbits
        period = 1.0
        x0_list = self.approximate_fixed_points()
        print("x0_list = \n{}".format(x0_list))
        self.find_stable_orbits(x0_list,period)
        return
    def drift_fun(self,x):
        q = self.q
        Nx = len(x)
        b = np.zeros((Nx,self.state_dim))
        b[:,0] = 1/q['tau']*(x[:,0] - x[:,0]**3 + q['kappa'] + q['lam']*np.cos(2*np.pi*x[:,1]))
        b[:,1] = 1.0
        #for d in range(1,self.state_dim):
        #    b[:,d] = -1/q['tau']*x[:,d]
        return b
    def drift_jacobian_fun(self,x):
        # single x
        bx = np.zeros((self.state_dim,self.state_dim))
        bx[0,0] = 1/q['tau']*(1 - 3*x[0]**2)
        bx[0,1] = 1/q['tau']*(-2*np.pi*q['lam']*np.sin(2*np.pi*x[:,1]))
        return bx
    def diffusion_fun(self,x):
        q = self.q
        Nx = len(x)
        wdot = np.random.randn(self.state_dim*Nx).reshape((self.state_dim,Nx))
        sig = q['sigma']*np.array([[1,0],[0,0]])
        sw = (sig.dot(wdot)).T
        return sw
    def diffusion_mat(self,x):
        return q['sigma']*np.array([[1,0],[0,0]])
    def tpt_observables(self,x):
        cvx = x
        cvx[:,1] = cvx[:,1] % 1.0 # Is this enough to periodize? 
        if cvx.shape[1] != self.tpt_obs_dim:
            sys.exit("DOH! tpt_observables output does not match expected dimension")
        return x
    def create_tpt_damage_functions(self):
        # A dictionary of lambda functions of interest to be integrated along reactive trajectories (or just forward-reactive or backward-reactive).
        q = self.q
        self.dam_dict = {
                'one': {
                    'pay': lambda x: np.ones(len(x)),
                    'name_fwd': "T+", #r"$\tau^+$",
                    'name_bwd': "T-", #r"$\tau^-$",
                    'name_full': "T", #r"$\tau^+-\tau^-$",
                    'abb_fwd': 't+',
                    'abb_bwd': 't-',
                    'abb_full': 'tfull',
                    'units': 1.0,
                    'unit_symbol': "",
                    'logscale': True,
                    },
                'potential': {
                    'pay': lambda x: 1/q['tau']*(-x[:,0]**2/2 + x[:,0]**4/4 - (q['kappa'] + q['lam']*np.cos(2*np.pi*x[:,1]))*x[:,0]),
                    'name_fwd': "V+", #r"$\int_0^{\tau^+}V(X(r))dr$",
                    'name_bwd': "V-", #r"$\int_{\tau^-}^0V(X(r))dr$",
                    'name_full': "V", #r"$\int_{\tau^-}^{\tau^+}V(X(r))dr$",
                    'abb_fwd': 'V+',
                    'abb_bwd': 'V-',
                    'abb_full': 'Vfull',
                    'units': 1.0,
                    'unit_symbol': '',
                    'logscale': False,
                    },
                }
        return
    def approximate_fixed_points(self):
        xst_approx = np.zeros((2,self.state_dim))
        xst_approx[0,0] = -1.0
        xst_approx[1,0] = 1.0
        return xst_approx
    def find_stable_orbits(self,x0_list,period):
        # One starts close to (1,0) and the other to (-1,0)
        #ta,xa = rk4(drift,np.array([[-1.0,0.0]]),0,10,0.001,q)
        #tb,xb = rk4(drift,np.array([[1.0,0.0]]),0,10,0.001,q)
        self.period = period
        print("x0_list.shape = {}".format(x0_list.shape))
        tmax = 100
        Nt = int(tmax/self.dt_sim) + 1
        tmax = (Nt-1)*self.dt_sim
        t = np.linspace(0,tmax,Nt)
        x = self.integrate_euler_maruyama(x0_list,t,stochastic_flag=False)
        print("x.shape = {}".format(x.shape))
        # Find which orbits end close to (1,0) and (-1,0)
        a_ends = np.where(x[-1,:,0] < 0)[0]
        b_ends = np.where(x[-1,:,0] > 0)[0]
        if len(a_ends) == 0 or len(b_ends) == 0: 
            sys.exit("PROBLEMO! Only one side found")
        xa = x[:,a_ends[0],:]
        xb = x[:,b_ends[0],:]
        print("xa.shape = {}".format(xa.shape))
        num_periods = t[-1] // self.period
        t0 = (num_periods-2)*self.period
        t1 = (num_periods-1)*self.period
        ti1 = np.argmin(np.abs(t-(t1+0.1*self.period)))
        ti0 = np.argmin(np.abs(t-(t0-0.1*self.period)))
        print("t0 = {}, t1 = {}".format(t[ti0],t[ti1]))
        alpha0 = interp1d(t[ti0:ti1]-t0,xa[ti0:ti1,0]) #,axis=0)
        self.alpha = lambda t: alpha0(t % self.period)
        beta0 = interp1d(t[ti0:ti1]-t0,xb[ti0:ti1,0]) #,axis=0)
        self.beta = lambda t: beta0(t % self.period)
        return
    def adist(self,cvx):
        # A whole list of (x,t) pairs
        cva = self.alpha(cvx[:,1])
        print("In adist. cva: min={}, max={}".format(np.min(cva),np.max(cva)))
        da = np.maximum(0, cvx[:,0]-cva)
        return da
    def bdist(self,cvx):
        # A whole list of (x,t) pairs
        cvb = self.beta(cvx[:,1])
        print("In bdist. cvb: min={}, max={}".format(np.min(cvb),np.max(cvb)))
        db = np.maximum(0, cvb-cvx[:,0])
        return db
    def set_param_folder(self):
        self.param_foldername = ("tau{}_kappa{}_lam{}_sigma{}_statedim{}".format(self.q['tau'],self.q['kappa'],self.q['lam'],self.q['sigma'],self.state_dim)).replace('.','p')
        return
    def regression_features(self,cvx):
        return cvx

def default_parameters():
    q = {
            'state_dim': 3,
            'tau': 0.25,
            'kappa': 0.00,
            'sigma': 1.0,
        }
    return q


