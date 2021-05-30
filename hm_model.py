
# This is where the Holton-Mass model is specified
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

class HoltonMassModel(Model):
    def __init__(self,hB_d=38.5,du_per_day=1.0,dh_per_day=0.0,ref_alt=30.0,abdefdim=75):
        self.ref_alt = ref_alt # in kilometers
        q = {
           'rad': 6370.0e3, 'day': 24*3600.0, 'g': 9.82, 'phi0': np.pi/3, 
           'sx': 2, 'sy': 3, 'zB_d': 0.0, 'zT_d': 70.0e3, 'H': 7.0e3, 
           'Omega': 2*np.pi/(24*3600), 'Nsq_d': 4.0e-4, 'ideal_gas_constant': 8.314,
           'eps': 8.0/(3*np.pi), 'UR_0_d': 10.0, 'gamma': 1.5, 'hB_d': hB_d, 
           'nfreq': 3, 'Nz': 26, 'length': 2.5e5, 'time': 24*3600.0,
           'du_per_day': du_per_day, 'dh_per_day': dh_per_day, 'dt_sim': 0.005,
           #'sig_u': 0.47, 'sig_h': 0.01, #'sig_ratio': 1e-2, 
        }
        self.q = self.initialize_params(q)
        # E[(dU)^2]/dt = (du_perday m/s)^2/day
        # E[(dpsi)^2]/dt = (g/f0*dh_perday m^2/s)^2/day
        #sig_h is the strength of forcing on the geopotential height, so 
        #forcing on the streamfunction is sig_psi = g*sig_h/f0
        self.abdefdim = abdefdim
        self.state_dim = 3*(self.q['Nz']-1)
        tpt_obs_dim = self.state_dim # We're dealing with full state here
        self.noise_rank = self.q['nfreq']
        self.dt_sim = self.q['dt_sim']
        parallel_sim_limit = 10000
        nshort_per_file_limit = 100000
        super().__init__(3*(self.q['Nz']-1),q['nfreq'],q['dt_sim'],tpt_obs_dim,parallel_sim_limit,nshort_per_file_limit)
        x0_list = self.approximate_fixed_points()
        self.find_fixed_points(tmax=600)
        self.tpt_obs_xst = self.tpt_observables(self.xst)
        return
    def initialize_params(self,q):
        #The z points will be (0, dz, 2*dz, ..., (Nz-1)*dz, Nz*dz). Therefore there are 
        #Nz+1 variables (some of them actually known) and dz=(zT-zB)/Nz
        ref_alt = self.ref_alt
        n = q['Nz']+1 #Number of points including boundaries
        q['state_dim'] = 3*(q['Nz']-1)
        #Dimensional variables
        q['f0_d'] = 2*q['Omega']*np.sin(q['phi0'])
        q['beta_d'] = 2*q['Omega']*np.cos(q['phi0'])/q['rad']
        q['k_d'] = q['sx']/(q['rad']*np.cos(q['phi0']))
        q['l_d'] = q['sy']/q['rad']
        q['dz_d'] = (q['zT_d']-q['zB_d'])/q['Nz']
        q['z_d'] = np.linspace(q['zB_d'],q['zT_d'],q['Nz']+1)
        q['alpha_d'] = (1.5 + np.tanh((q['z_d']/1000-25)/7.0))*1e-6
        q['alpha_z_d'] = 1e-6/7000 * 1.0/np.cosh((q['z_d']/1000-25)/7.0)**2
        q['UR_d'] = q['UR_0_d'] + q['gamma']*q['z_d']/1000
        q['UR_z_d'] = q['gamma']/1000*np.ones(n)
        q['UR_zz_d'] = np.zeros(n)
        q['Psi0_d'] = q['g']*q['hB_d']/q['f0_d']
        q['fn'] = q['f0_d']**2/q['Nsq_d']
        q['lap_d'] = q['k_d']**2 + q['l_d']**2 + 1.0/(4*q['H']**2)*q['fn']
        #Dimensionless variables
        q['Gsq'] = 1.0/(q['length']**2/q['H']**2*q['fn'])
        q['k'] = q['k_d']*q['length']
        q['l'] = q['l_d']*q['length']
        q['dz'] = q['dz_d']/q['H']
        q['zT'] = q['zT_d']/q['H']
        q['zB'] = q['zB_d']/q['H']
        q['z'] = q['z_d']/q['H']
        q['beta'] = q['beta_d']*q['time']*q['length']
        q['alpha'] = q['alpha_d']*q['time']
        q['alpha_z'] = q['alpha_z_d']*q['time']*q['H']
        q['UR'] = q['UR_d']*q['time']/q['length']
        q['UR_0'] = q['UR_0_d']*q['time']/q['length']
        q['UR_z'] = q['UR_z_d']*q['time']*q['H']/q['length']
        q['UR_zz'] = q['UR_zz_d']*q['time']*q['H']**2/q['length']
        q['lap'] = q['Gsq']*(q['k']**2 + q['l']**2) + 1/4.0
        q['Psi0'] = q['Psi0_d']*q['time']/q['length']**2
        #Some handy shortcuts
        q['zi'] = np.argmin(np.abs(q['z_d'][1:-1]/1000-self.ref_alt))
        q['states'] = ['r','v']
        #Noise
        q['sig_u'] = q['du_per_day']*np.sqrt(q['time']**3/q['length']**2/(3600.0*24))
        n = q['Nz']-1
        sigmat = np.zeros((3*n,q['nfreq'])) 
        # Wdot = np.zeros(3*n)
        for k in range(q['nfreq']):
            sinkz_shift = np.sin((k+0.5)*np.pi*q['z']/q['zT'])[1:-1]
            sigmat[2*n:,k] = q['sig_u']*sinkz_shift
            # Wdot[[k,n+k,2*n+k]] = mag[k]
        q['sig_mat'] = sps.csr_matrix(sigmat)
        # Left-hand inversion operator
        Lpsi = np.zeros((n,n))
        Lu = np.zeros((n,n))
        Lpsi[np.arange(n),np.arange(n)] = -(q['Gsq']*(q['k']**2+q['l']**2) + 0.25) - 2.0/q['dz']**2
        Lpsi[np.arange(n-1),np.arange(1,n)] = 1.0/q['dz']**2
        Lpsi[np.arange(1,n),np.arange(n-1)] = 1.0/q['dz']**2
        Lu[np.arange(n),np.arange(n)] = -q['Gsq']*q['l']**2 - 2.0/q['dz']**2
        Lu[np.arange(n-1),np.arange(1,n)] = -1.0/(2*q['dz']) + 1.0/q['dz']**2
        Lu[np.arange(1,n),np.arange(n-1)] = 1.0/(2*q['dz']) + 1.0/q['dz']**2
        # Boundary!
        Lu[n-1,n-2] = 2.0/(3*q['dz']) + 2.0/(3*q['dz']**2)
        Lu[n-1,n-1] = -q['Gsq']*q['l']**2 - 2.0/(3*q['dz']) - 2.0/(3*q['dz']**2)
        # Invert
        q['Lpsi_inv'] = sps.csr_matrix(np.linalg.inv(Lpsi))
        q['Lu_inv'] = sps.csr_matrix(np.linalg.inv(Lu))
        dz = q['dz']
        self.Xz_vec,self.Xz_mat = D1mat(n,dz,q['Psi0'],'dirichlet',0,'dirichlet')
        self.Xzz_vec,self.Xzz_mat = D2mat(n,dz,q['Psi0'],'dirichlet',0,'dirichlet')
        self.Yz_vec,self.Yz_mat = D1mat(n,dz,0,'dirichlet',0,'dirichlet')
        self.Yzz_vec,self.Yzz_mat = D2mat(n,dz,0,'dirichlet',0,'dirichlet')
        self.Uz_vec,self.Uz_mat = D1mat(n,dz,q['UR_0'],'dirichlet',q['UR_z'][-1],'neumann')
        self.Uzz_vec,self.Uzz_mat = D2mat(n,dz,q['UR_0'],'dirichlet',q['UR_z'][-1],'neumann')
        # Get constant and linear parts first
        self.Jlin = np.zeros((3*n,3*n))
        # Re(Psi)
        a = q['alpha'][1:-1]
        az = q['alpha_z'][1:-1]
        self.Jlin[:n,:n] += np.diag(a/4-az/2) - (az*self.Xz_mat.T + a*self.Xzz_mat.T).T
        self.Jlin[:n,n:2*n] += q['Gsq']*q['k']*q['beta']*np.eye(n)
        # Im(Psi)
        self.Jlin[n:2*n,n:2*n] += np.diag(a/4-az/2) - (az*self.Yz_mat.T + a*self.Yzz_mat.T).T
        self.Jlin[n:2*n,:n] += -q['Gsq']*q['k']*q['beta']*np.eye(n)
        # U
        self.Jlin[2*n:3*n,2*n:3*n] = -((az-a)*self.Uz_mat.T + a*self.Uzz_mat.T).T
        return q
    def drift_jacobian_fun(self,x):
        # x is just a single instance
        q = self.q
        n = q['Nz']-1
        a = q['alpha'][1:-1]
        az = q['alpha_z'][1:-1]
        z = q['z'][1:-1]
        dz = q['dz']
        J = self.Jlin.copy()
        X = x[0:n]
        Y = x[n:2*n]
        U = x[2*n:3*n]
        Xz = self.Xz_mat.dot(X) + self.Xz_vec
        Xzz = self.Xzz_mat.dot(X) + self.Xzz_vec
        Yz = self.Yz_mat.dot(Y) + self.Yz_vec
        Yzz = self.Yzz_mat.dot(Y) + self.Yzz_vec
        Uz = self.Uz_mat.dot(U) + self.Uz_vec
        Uzz = self.Uzz_mat.dot(U) + self.Uzz_vec
        # The matrix will consist of nine blocks
        # Re(Psi)
        J[:n,n:2*n] += -q['k']*q['eps']*np.diag((q['k']**2*q['Gsq']+0.25)*U - Uz + Uzz) + q['k']*q['eps']*(U*self.Yzz_mat.T).T
        J[:n,2*n:3*n] += -(q['k']*q['eps']*Y*((q['k']**2*q['Gsq']+0.25)*np.eye(n) - self.Uz_mat + self.Uzz_mat).T).T + q['k']*q['eps']*np.diag(Yzz)
        # Im(Psi)
        J[n:2*n,:n] += q['k']*q['eps']*np.diag((q['k']**2*q['Gsq']+0.25)*U - Uz + Uzz) - q['k']*q['eps']*(U*self.Xzz_mat.T).T
        J[n:2*n,2*n:3*n] += (q['k']*q['eps']*X*((q['k']**2*q['Gsq']+0.25)*np.eye(n) - self.Uz_mat + self.Uzz_mat).T).T - q['k']*q['eps']*np.diag(Xzz)
        # U
        J[2*n:3*n,:n] += q['eps']*q['k']*q['l']**2/2*np.diag(np.exp(z)).dot((-Yzz*np.eye(n).T + Y*self.Xzz_mat.T).T)
        J[2*n:3*n,n:2*n] += q['eps']*q['k']*q['l']**2/2*np.diag(np.exp(z)).dot((-X*self.Yzz_mat.T + Xzz*np.eye(n).T).T)
        # Now invert
        J[:n,:] = q['Lpsi_inv'].dot(J[:n,:])
        J[n:2*n,:] = q['Lpsi_inv'].dot(J[n:2*n,:])
        J[2*n:3*n,:] = q['Lu_inv'].dot(J[2*n:3*n,:])
        return J
    def drift_fun(self,x):
        q = self.q
        n = q['Nz']-1
        Nx = len(x)
        # Build the right-hand side
        rhs = np.zeros((Nx,3*n))
        X = x[:,0:n]
        Y = x[:,n:2*n]
        U = x[:,2*n:3*n]
        Xz = first_derivative(X,q['Psi0'],0,q['dz'])
        Xzz = second_derivative(X,q['Psi0'],0,q['dz'])
        Yz = first_derivative(Y,0,0,q['dz'])
        Yzz = second_derivative(Y,0,0,q['dz'])
        U_upper = 1.0/3*(4*U[:,-1] - U[:,-2] + 2*q['dz']*q['UR_z'][-1])
        Uz = first_derivative(U,q['UR_0'],U_upper,q['dz'])
        Uzz = second_derivative(U,q['UR_0'],U_upper,q['dz'])
        # Re(Psi)
        rhs[:,:n] = (q['alpha'][1:-1]/4 - q['alpha_z'][1:-1]/2)*X + q['Gsq']*q['k']*q['beta']*Y - q['alpha_z'][1:-1]*Xz - q['alpha'][1:-1]*Xzz
        rhs[:,:n] += (-q['k']*q['eps']*Y)*((q['k']**2*q['Gsq']+0.25)*U - Uz + Uzz)
        rhs[:,:n] += q['k']*q['eps']*Yzz*U
        # Im(Psi)
        rhs[:,n:2*n] = (q['alpha'][1:-1]/4 - q['alpha_z'][1:-1]/2)*Y - q['Gsq']*q['k']*q['beta']*X - q['alpha_z'][1:-1]*Yz - q['alpha'][1:-1]*Yzz
        rhs[:,n:2*n] += q['k']*q['eps']*X*((q['k']**2*q['Gsq']+0.25)*U - Uz + Uzz)
        rhs[:,n:2*n] += -q['k']*q['eps']*Xzz*U
        # U
        rhs[:,2*n:3*n] = ((q['alpha_z'][1:-1]-q['alpha'][1:-1])*q['UR_z'][1:-1]) 
        rhs[:,2*n:3*n] += -(q['alpha_z'][1:-1]-q['alpha'][1:-1])*Uz - q['alpha'][1:-1]*Uzz
        rhs[:,2*n:3*n] += q['eps']*q['k']*q['l']**2/2*np.exp(q['z'][1:-1])*(-X*Yzz + Y*Xzz)
        # Now invert the Laplacian on the left
        #xdot = q['L_inv'].dot(rhs.T).T
        xdot = np.zeros((Nx,3*n))
        xdot[:,:n] = (q['Lpsi_inv'].dot(rhs[:,:n].T)).T 
        xdot[:,n:2*n] = (q['Lpsi_inv'].dot(rhs[:,n:2*n].T)).T 
        xdot[:,2*n:3*n] = (q['Lu_inv'].dot(rhs[:,2*n:3*n].T)).T 
        return xdot
    def diffusion_fun(self,x):
        wdot = np.random.randn(len(x)*self.q['nfreq']).reshape((self.q['nfreq'],len(x)))
        diff_term = (self.q['sig_mat'].dot(wdot)).T
        return diff_term
    def diffusion_mat(self,x):
        return self.q['sig_mat']
    def tpt_observables(self,x):
        return x # No reduction happening here
    def adist(self,cvx):
        cva = self.tpt_observables(self.xst[0])
        if self.abdefdim == 75:
            radius_a = 8.0
            da = np.sqrt(np.sum((cvx - cva)**2, 1))
            return np.maximum(0, da-radius_a)
        else:
            n = self.q['Nz']-1
            zi = self.q['zi']
            da = cva[2*n+zi] - cvx[:,2*n+zi]
            return np.maximum(da,0)
    def bdist(self,cvx):
        cvb = self.tpt_observables(self.xst[1])
        if self.abdefdim == 75:
            radius_b = 30.0 #20.0
            db = np.sqrt(np.sum((cvx - cvb)**2, 1))
            return np.maximum(0, db-radius_b)
        else:
            n = self.q['Nz']-1
            zi = self.q['zi']
            db = cvx[:,2*n+zi] - cvb[2*n+zi]
            return np.maximum(db,0)
    def set_param_folder(self):
        self.param_foldername = ("du{}_h{}".format(self.q['du_per_day'],self.q['hB_d'])).replace(".","p")
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
        ax[0,0].set_title(r"$A\to B$",fontdict=bigfont)
        ax[0,1].set_title(r"$B\to A$",fontdict=bigfont)
        ax[1,0].set_title("%s"%(funlib[fun_name]["name"]),fontdict=bigfont)
        ax[1,1].set_title("%s"%(funlib[fun_name]["name"]),fontdict=bigfont)
        ax[2,0].set_title(r"$\delta U(t)$",fontdict=bigfont)
        ax[2,1].set_title(r"$\delta U(t)$",fontdict=bigfont)
        ax[0,0].set_ylabel("%s at z=%d km (%s)"%(funlib[fun_name]["name"],self.ref_alt,funlib[fun_name]["unit_symbol"]),fontdict=bigfont)
        ax[1,0].set_ylabel(r"$z\,(km)$",fontdict=bigfont)
        ax[2,0].set_ylabel(r"$z\,(km)$",fontdict=bigfont)
        # Tick labels
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i,j].tick_params(axis='both',labelsize=30)
        # Save
        fig.savefig(join(physical_param_folder,"fw_plot_%s"%fun_name))
        plt.close(fig)
        return 
    def create_tpt_damage_functions(self):
        # A dictionary of lambda functions of interest to be integrated along reactive trajectories (or just forward-reactive or backward-reactive).
        q = self.q
        funlib = self.observable_function_library()
        self.corr_dict = {
                'one': {
                    'pay': lambda x: np.ones(len(x)),
                    'name': '1',
                    },
                'heatflux': {
                    'pay': funlib['vTref']['fun'], 
                    'name': funlib['vTref']['name'],
                    },
                'heatflux_g5em5': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 5e-5),
                    'name':  r"$1\{$%s$ > 5\times10^5\ K\cdot m/s\}$"%funlib['vTref']['name'],
                    },
                'magref': {
                    'pay': funlib['magref']['fun'],
                    'name': funlib['magref']['name'],
                    },
                'magref_g1e7': {
                    'pay': lambda x: 1.0*(funlib['magref']['fun'](x)*funlib['magref']['units'] > 1e7),
                    'name': r"$1\{$%s$ > 10^7\ m^2/s\}$"%funlib['magref']['name'],
                    },
                'Uref': {
                    'pay': funlib["Uref"]["fun"],
                    'name': funlib['Uref']['name'],
                    },
                'Uref_l0': {
                    'pay': lambda x: 1.0*(funlib['Uref']['fun'](x) < 0),
                    'name': r"$1\{$%s$ < 0\}$"%funlib['Uref']['name'],
                    },
                'Uref_ln20': {
                    'pay': lambda x: 1.0*(funlib['Uref']['fun'](x)*funlib['Uref']['units'] < -20),
                    'name':  r"$1\{$%s$ < -20\ m/s\}$"%funlib['Uref']['name'],
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
                    'unit_symbol': "",
                    'unit_symbol_t': "days",
                    'logscale': True,
                    'pay_symbol': "1",
                    },
                'heatflux': {
                    'pay': self.fun_at_level(funlib["vT"]["fun"], self.ref_alt),
                    'name': 'Heat flux (%.0f\\text{ km})'%self.ref_alt,
                    'name_fwd': "\\int_0^{\\tau^+}\\overline{v'T'}(%.0f\\ km)dt"%self.ref_alt, 
                    'name_bwd': "\\int_{\\tau^-}^0\\overline{v'T'}(%.0f\\ km)dt"%self.ref_alt, #r"$\tau^-$",
                    'name_full': "\\int_{\\tau^-}^{\\tau^+}\\overline{v'T'}(%.0f\\ km)dt"%self.ref_alt, #r"$\tau^+-\tau^-$",
                    'abb_fwd': 'vT+',
                    'abb_bwd': 'vT-',
                    'abb_full': 'vTfull',
                    'units': funlib["vT"]["units"],
                    'unit_symbol': "\\mathrm{K}\\cdot\\mathrm{m/s}", 
                    'unit_symbol_t': "\\mathrm{K}\\cdot\\mathrm{m}", # This is the unit after being multiplied by time 
                    'logscale': False,
                    'pay_symbol': "\\overline{v'T'}(%.0f\\mathrm{ km})"%self.ref_alt,
                    },
                #'heatfluxint': {
                #    'pay': self.fun_at_level(funlib["vTint"]["fun"], q['z_d'][-2]/1000),
                #    'name': 'Integrated Heat flux',
                #    'name_fwd': "\\int_0^{\\tau^+}\\int\\rho\\overline{v'T'}dzdt", 
                #    'name_bwd': "\\int_{\\tau^-}^0\\int\\rho\\overline{v'T'}dt", #r"$\tau^-$",
                #    'name_full': "\\int_{\\tau^-}^{\\tau^+}\\int\\rho\\overline{v'T'}dzdt", #r"$\tau^+-\tau^-$",
                #    'abb_fwd': 'vTi+',
                #    'abb_bwd': 'vTi-',
                #    'abb_full': 'vTifull',
                #    'units': funlib["vTint"]["units"],
                #    'unit_symbol': funlib["vTint"]["unit_symbol"],
                #    'logscale': False,
                #    'pay_symbol': "\\int\\rho^{-1}\\overline{v'T'}dz",
                #    },
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
    def product_rule_z(self,X,lower,upper,k):
        q = self.q
        # Compute the vertical derivatives of Xe^(z/2) (units are nondimensional)
        n = q['Nz']-1
        Xz = first_derivative(X,lower,upper,q['dz'])
        if k == 1:
            Xder = (Xz + 0.5*X)*np.exp(q['z'][1:-1]/2)
        elif k == 2:
            Xzz = second_derivative(X,lower,upper,q['dz'])
            Xder = (Xzz + Xz + 0.25*X)*np.exp(q['z'][1:-1]/2)
        return Xder
    def approximate_fixed_points(self):
        q = self.q
        n = q['Nz']-1
        x = np.zeros((2,3*n))
        x[:,:n] = q['Psi0']*(1 - q['z'][1:-1]/q['zT'])
        x[:,n:2*n] = 0.0
        x[0,2*n:3*n] = q['UR'][1:-1]
        kneez = np.argmin(np.abs(q['z_d'][1:-1] - 35.0e3))
        x[1,2*n:3*n] = np.maximum(q['UR_0'], q['UR_0'] + q['UR_z'][-1]*(q['z'][1:-1] - q['z'][kneez]))
        return x
    def enstrophy(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        # Compute all the necessary correlations
        X,Y = x[:,:n],x[:,n:2*n]
        X1,Y1 = self.product_rule_z(X,q['Psi0'],0,1),self.product_rule_z(Y,0,0,1)
        X2,Y2 = self.product_rule_z(X,q['Psi0'],0,2),self.product_rule_z(Y,0,0,2)
        qpsq = (q['k']**2 + q['l']**2)*(X**2 + Y**2)
        qpsq += -1/q['Gsq']*(q['k']**2 + q['l']**2) * (
                X*X2 + Y*Y2 - X*X1 - Y*Y1)
        qpsq += 1/q['Gsq']**2 * (X2**2 + Y2**2 - 2*(X1*X2 + Y1*Y2) + X1**2 + Y1**2)
        qpsq *= 0.5*17/35
        return qpsq
    def meridional_heat_flux(self,x):
        q = self.q
        # Compute the meridional heat flux, perhaps of the whole timeseries
        n = q['Nz']-1
        Nt = len(x)
        heat_flux = np.ones([Nt,q['Nz']-1])
        heat_flux *= q['k'] 
        #heat_flux *= q['k_d']*q['H']*q['f0_d']/(2*q['ideal_gas_constant'])
        heat_flux *= np.exp(q['z'][1:-1])*17.0/35
        # Now it has to be multiplied by vertical derivatives
        Xz = first_derivative(x[:,:n],q['Psi0'],0,q['dz']) 
        Yz = first_derivative(x[:,n:2*n],0,0,q['dz'])
        heat_flux *= (x[:,:n]*Yz - x[:,n:2*n]*Xz)
        return heat_flux
    def integrated_meridional_heat_flux(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        heat_flux = np.ones([Nt,n+1])
        heat_flux *= q['k'] #q['k_d']*q['H']*q['f0_d']/(2*q['ideal_gas_constant'])
        #heat_flux *= np.exp(q['z'][:-1])*17.0/35
        # Now it has to be multiplied by vertical derivatives
        Xz = first_derivative(x[:,:n],q['Psi0'],0,q['dz']) 
        Yz = first_derivative(x[:,n:2*n],0,0,q['dz'])
        Yz0 = (4*x[:,2*n] - x[:,2*n+1])/(2*q['dz'])
        heat_flux[:,1:] *= (x[:,:n]*Yz - x[:,n:2*n]*Xz)
        heat_flux[:,0] *= q['Psi0']*Yz0
        heat_flux *= np.exp(-q['z'][:-1])
        ivt = np.zeros((Nt,n))
        ivt[:,0] = 0.5*(heat_flux[:,0] + heat_flux[:,1])*q['dz']
        for i in range(1,n):
            ivt[:,i] = ivt[:,i-1] + 0.5*(heat_flux[:,i] + heat_flux[:,i+1])*q['dz']
        return ivt
    def background_pv_gradient(self,x):
        q = self.q
        # Compute the meridional gradient of zonal-mean potential vorticity
        #rho0 = 0.41
        n = q['Nz']-1
        Nt = len(x)
        U = x[:,2*n:3*n]
        Utop = (2*q['dz']*q['UR_z'][-1] - U[:,-2] + 4*U[:,-1])/3
        Uz = first_derivative(U,q['UR_0'],Utop,q['dz'])
        #Uz[:,-1] = q['UR_z'][-1]
        Uzz = second_derivative(U,q['UR_0'],Utop,q['dz'])
        #Uzz[:,-1] = (2*U[:,-2] + 2*q['dz']*q['UR_z'][-1] - 2*U[:,-1])/q['dz']**2
        qbar_grad = q['beta'] + q['l']**2*U*3.0/8 + 1/q['Gsq']*(Uz*3.0/8 - Uzz*3.0/8)
        #qbar_grad *= 1.0/(q['length']*q['time'])
        return qbar_grad
    def epflux_z(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        pv_flux = np.ones([Nt,q['Nz']-1])
        pv_flux *= q['k']*17.0/35*np.exp(q['z'][1:-1])/2 # Need to put in rho0. 17/35 is the meridional average of sin^2*(3y)
        # Now it has to be multiplied by vertical derivatives
        Xzz = second_derivative(x[:,:n],q['Psi0'],0,q['dz'])
        Yzz = second_derivative(x[:,n:2*n],0,0,q['dz'])
        pv_flux *= (x[:,:n]*Yzz - x[:,n:2*n]*Xzz)
        dens = np.exp(-q['z'][1:-1])
        #pv_flux *= q['length']**3/(q['H']**2*q['time']**2)
        return pv_flux*dens
    def meridional_pv_flux(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        pv_flux = np.ones([Nt,q['Nz']-1])
        pv_flux *= q['k']*17.0/35*np.exp(q['z'][1:-1])/2 # Need to put in rho0. 17/35 is the meridional average of sin^2*(3y)
        # Now it has to be multiplied by vertical derivatives
        Xzz = second_derivative(x[:,:n],q['Psi0'],0,q['dz'])
        Yzz = second_derivative(x[:,n:2*n],0,0,q['dz'])
        pv_flux *= (x[:,:n]*Yzz - x[:,n:2*n]*Xzz)
        #pv_flux *= q['length']**3/(q['H']**2*q['time']**2)
        return pv_flux
    def angular_displacement_times_magnitude(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        phase = np.zeros((Nt,n))
        phase0 = np.zeros(Nt)
        for i in range(1,n):
            phase[:,i] = np.arctan2(x[:,n+i],x[:,i])
            phase[:,i] += 2*np.pi*(np.round((phase0-phase[:,i])/(2*np.pi)).astype(int))
            # Edge cases
            #idx = np.where((phase0<-np.pi/2)*(phase[:,i]>np.pi/2))[0]
            #phase[idx,i] -= 2*np.pi
            #idx = np.where((phase0>np.pi/2)*(phase[:,i]<-np.pi/2))[0]
            #phase[idx,i] += 2*np.pi
            phase0 = phase[:,i]
        ang_disp_mag = -phase/(q['sx'])
        return ang_disp_mag
    def wave_transience(self,x):
        q = self.q
        # The wave transience term (but not dividided by dq/dy)
        n = q['Nz']-1
        Nt = len(x)
        X,Y = x[:,:n],x[:,n:2*n]
        X1,Y1 = product_rule_z(X,q['Psi0'],0,q,1),product_rule_z(Y,0,0,q,1)
        X2,Y2 = product_rule_z(X,q['Psi0'],0,q,2),product_rule_z(Y,0,0,q,2)
        a = q['alpha'][1:-1]
        az = q['alpha_z'][1:-1]
        wt = 1/q['Gsq']*((q['k']**2+q['l']**2)*(X*X2 + Y*Y2)
                + (az - a)*(X*X1 + Y*Y1))
        wt += 1/q['Gsq']**2 * (-a*(X2**2 + Y2**2)
                + (a - az)*(X2*X1 + Y2*Y1))
        wt *= 0.5*17/35
        return wt
    def regression_features0(self,x):
        funlib = self.observable_function_library()
        n = self.q['Nz']-1
        Nx = len(x)
        lass = np.zeros((Nx,6*n))
        lass[:,:n] = x[:,:n]
        lass[:,n:2*n] = funlib["repsi_z"]["fun"](x)
        lass[:,2*n:3*n] = x[:,n:2*n]
        lass[:,3*n:4*n] = funlib["impsi_z"]["fun"](x)
        lass[:,4*n:5*n] = x[:,2*n:3*n]
        lass[:,5*n:6*n] = funlib["U_z"]["fun"](x)
        return lass
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
        n = q['Nz'] - 1
        funs = {
                "U":
                {"fun": lambda X: X[:,2*n:3*n],
                 "name":r"$U$",
                 "units": q['length']/q['time'],
                 "unit_symbol": r"m/s"
                 },
                "Uref":
                {"fun": lambda X: X[:,2*n+q['zi']],
                 "name":r"$U$(%.0f km)"%self.ref_alt,
                 "units": q['length']/q['time'],
                 "unit_symbol": r"m/s"
                 },
                "U21p5":
                {"fun": lambda X: X[:,2*n+zlevel(21.5)],
                 "name":r"$U(21.5\,km)$",
                 "units": q['length']/q['time'],
                 "unit_symbol": r"m/s"
                 },
                "U13p5":
                {"fun": lambda X: X[:,2*n+zlevel(13.5)],
                 "name":r"$U(13.5\,km)$",
                 "units": q['length']/q['time'],
                 "unit_symbol": r"m/s"
                 },
                "U19":
                {"fun": lambda X: X[:,2*n+zlevel(19)],
                 "name":r"$U(19\,km)$",
                 "units": q['length']/q['time'],
                 "unit_symbol": r"m/s"
                 },
                "U67":
                {"fun": lambda X: X[:,2*n+zlevel(67)],
                 "name":r"$U(67\,km)$",
                 "units": q['length']/q['time'],
                 "unit_symbol": r"m/s"
                 },
                "U_z":
                {"fun": lambda X: first_derivative(X[:,2*n:3*n],q['UR_0'],1.0/3*(4*X[:,3*n-1] - X[:,3*n-2] + 2*q['dz']*q['UR_z'][-1]),q['dz']),
                 "name":r"$U_z$",
                 "units": q['length']/(q['H']*q['time']),
                 "unit_symbol": r"$s^{-1}$"
                 },
                "U_zz":
                {"fun": lambda X: second_derivative(X[:,2*n:3*n],q['UR_0'],1.0/3*(4*X[:,3*n-1] - X[:,3*n-2] + 2*q['dz']*q['UR_z'][-1]),q['dz']),
                 "name":r"$U_{zz}$",
                 "units": q['length']/(q['H']**2*q['time']),
                 "unit_symbol": r"$m^{-1}s^{-1}$"
                },
                "imhf": 
                {"fun": lambda X: integrated_meridional_heat_flux(X,q), 
                 "name": r"$\int_0^zr^2\phi_z\ dz$"
                 },
                "admag": 
                {"fun": lambda X: angular_displacement_times_magnitude(X,q), 
                 "name": r"$|\Psi|\lambda$"
                 },
                "mag": 
                {"fun": lambda X: np.sqrt(X[:,:n]**2+X[:,n:2*n]**2),
                 "name": r"$|\Psi|$",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"m$^2/$s",
                 },
                "magref": 
                {"fun": lambda X: np.sqrt(X[:,q['zi']]**2+X[:,n+q['zi']]**2),
                 "name": r"$|\Psi|$(%.0f km)"%self.ref_alt,
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"m$^2/$s",
                 },
                "mag21p5": 
                {"fun": lambda X: np.sqrt(X[:,zlevel(21.5)]**2+X[:,n+zlevel(21.5)]**2),
                 "name": r"$|\Psi|$(21.5 km)",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"m$^2/$s",
                 },
                "mag13p5": 
                {"fun": lambda X: np.sqrt(X[:,zlevel(13.5)]**2+X[:,n+zlevel(13.5)]**2),
                 "name": r"$|\Psi|$(13.5 km)",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"m$^2/$s",
                 },
                "mag19": 
                {"fun": lambda X: np.sqrt(X[:,zlevel(19)]**2+X[:,n+zlevel(19)]**2),
                 "name": r"$|\Psi|$(19 km)",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"$m^2/s$",
                 },
                "repsi": 
                {"fun": lambda X: X[:,:n],
                 "name": r"$Re\{\Psi\}$",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"m$^2/$s",
                 },
                "repsiref": 
                {"fun": lambda X: X[:,q['zi']],
                 "name": r"$Re\{\Psi\}$(%.0f km)"%self.ref_alt,
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"m$^2/$s",
                 },
                "repsi_z": 
                {"fun": lambda X: first_derivative(X[:,:n],q['Psi0'],0,q['dz']),
                 "name": r"$Re\{\Psi_z\}$",
                 "units": q['length']**2/(q['H']*q['time']),
                 "unit_symbol": r"m/s",
                 },
                "repsi_zz": 
                {"fun": lambda X: second_derivative(X[:,:n],q['Psi0'],0,q['dz']),
                 "name": r"$Re\{\Psi_{zz}\}$",
                 "units": q['length']**2/(q['H']**2*q['time']),
                 "unit_symbol": r"s^{-1}$",
                 },
                "impsi": 
                {"fun": lambda X: X[:,n:2*n],
                 "name": r"$Im\{\Psi\}$",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"$m^2/s$",
                 },
                "impsi13p5": 
                {"fun": lambda X: X[:,n+zlevel(13.6)],
                 "name": r"$Im(\Psi)(13.5\ km)$",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"$m^2/s$",
                 },
                "impsi19": 
                {"fun": lambda X: X[:,n+zlevel(19)],
                 "name": r"$Im(\Psi)(19\ km)$",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"$m^2/s$",
                 },
                "impsi21p5": 
                {"fun": lambda X: X[:,n+zlevel(21.5)],
                 "name": r"$Im(\Psi)(21.5\ km)$",
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"$m^2/s$",
                 },
                "impsiref": 
                {"fun": lambda X: X[:,n+q['zi']],
                 "name": r"$Im(\Psi)(%.0f km)$"%self.ref_alt,
                 "units": q['length']**2/q['time'],
                 "unit_symbol": r"$m^2/s$",
                 },
                "impsi_z": 
                {"fun": lambda X: first_derivative(X[:,n:2*n],0,0,q['dz']),
                 "name": r"$Im(\Psi_z)$",
                 "units": q['length']**2/(q['H']*q['time']),
                 "unit_symbol": r"m/s",
                 },
                "impsi_zz": 
                {"fun": lambda X: second_derivative(X[:,n:2*n],0,0,q['dz']),
                 "name": r"$Im(\Psi_{zz})$",
                 "units": q['length']**2/(q['H']**2*q['time']),
                 "unit_symbol": r"$s^{-1}$",
                 },
                "vTint": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X),
                 "name": r"$\int_{0}^{z}e^{-z/H}\overline{v'T'}dz$",
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "vTint13p5": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,zlevel(13.5)],
                 "name": r"$\int_{0\ km}^{%.1f\ km}e^{-z/H}\overline{v'T'}dz$"%(13.5),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "vTint19": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,zlevel(19)],
                 "name": r"$\int_{0\ km}^{%.1f\ km}e^{-z/H}\overline{v'T'}dz$"%(19),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "vTint21p5": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,zlevel(21.5)],
                 "name": r"$\int_{0\ km}^{%.1f\ km}e^{-z/H}\overline{v'T'}dz$"%(21.5),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "vTintref": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,q['zi']],
                 "name": r"$\int_{0\ \mathrm{km}}^{%.0f\ \mathrm{km}}e^{-z/H}\overline{v'T'}dz$"%(self.ref_alt),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"K$\cdot$m$^2/$s",
                 },
                "vTinttop": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,-1],
                 "name": r"$\int_{z_b}^{%.0f}e^{-z/H}\overline{v'T'}dz$"%(q['z_d'][-2]/1000),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "vT": 
                {"fun": lambda X: self.meridional_heat_flux(X),
                 "name": r"$\overline{v'T'}$",
                 "units": q['H']*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"K$\cdot$m/s",
                 },
                "vTref": 
                {"fun": lambda X: self.meridional_heat_flux(X)[:,q['zi']],
                 "name": r"$\overline{v'T'}(%.0f km)$"%self.ref_alt,
                 "units": q['H']*q['f0_d']/(2*q['length']*q['ideal_gas_constant']),
                 "unit_symbol": r"$K\cdot m/s$",
                 },
                "dqdy":
                {"fun": lambda X: self.background_pv_gradient(X),
                 "name": r"$\partial_y\overline{q}$",
                 "units": 1.0/(q['length']*q['time']),
                 "unit_symbol": r"m$^{-1}$s$^{-1}$",
                 },
                "dqdymean":
                {"fun": lambda X: np.mean(self.background_pv_gradient(X),1),
                 "name": r"$\partial_y\overline{q}$ (z-mean)",
                 "units": 1.0/(q['length']*q['time']),
                 "unit_symbol": r"m$^{-1}$s$^{-1}$",
                 },
                "dqdyref":
                {"fun": lambda X: self.background_pv_gradient(X)[:,q['zi']],
                 "name": r"$\partial_y\overline{q} (%.0f km)$"%self.ref_alt,
                 "units": 1.0/(q['length']*q['time']),
                 "unit_symbol": r"$m^{-1}s^{-1}$",
                 },
                "epflux_z":
                {"fun": lambda X: self.epflux_z(X),
                 "name":r"$e^{-z/H}\overline{v'q'}$",
                 "units": q['length']**3/(q['H']**2*q['time']**2),
                 "unit_symbol": r"s$^{-1}$",
                 },
                "vq":
                {"fun": lambda X: self.meridional_pv_flux(X),
                 "name":r"$\overline{v'q'}$",
                 "units": q['length']**3/(q['H']**2*q['time']**2),
                 "unit_symbol": r"s$^{-1}$",
                 },
                "q2":
                {"fun": lambda X: self.enstrophy(X),
                 "name":r"$\overline{q'^2}$",
                 "units": 1/q['time']**2,
                 "unit_symbol": r"s$^{-2}$",
                 },
                "q2ref":
                {"fun": lambda X: self.enstrophy(X)[:,q['zi']],
                 "name":r"$\overline{q'^2} (%.0f km)$"%self.ref_alt,
                 "units": 1/q['time']**2,
                 "unit_symbol": r"$s^{-2}$",
                 },
                "q2mean":
                {"fun": lambda X: np.mean(self.enstrophy(X),1),
                 "name":r"$\overline{q'^2}$ (z-mean)",
                 "units": 1/q['time']**2,
                 "unit_symbol": r"$s^{-2}$",
                 },
                "wtran":
                {"fun": lambda X: wave_transience(X,q),
                 "name": r"$-\frac{f_0^2}{N^2}\overline{q'\rho_s^{-1}\partial_z(\alpha\rho_s\partial_z\psi')}$",
                 },
            }
        return funs
    def plot_snapshot(self,xt,suffix=""):
        q = self.q
        # Plot a latitude-height section of zonal wind and streamfunction phase. 
        n = q['Nz']-1
        def fmt(num,pos):
            return '{:.1f}'.format(num)
        fig,ax = plt.subplots(ncols=2,figsize=(12,6))
        y = np.linspace(30,90,50)
        yz,zy = np.meshgrid(y,q['z_d'][1:-1]/1000,indexing='ij')
        U = np.outer(np.sin(3*y*np.pi/180),xt[2*n:3*n])
        im = ax[0].contourf(yz,zy,U,cmap='seismic')
        ax[0].set_xlabel("Latitude",fontdict=bigfont)
        ax[0].set_ylabel("Height (km)",fontdict=bigfont)
        ax[0].set_title("Zonal wind {}".format(suffix),fontdict=bigfont)
        fig.colorbar(im,ax=ax[0],format=ticker.FuncFormatter(fmt),
                orientation='horizontal',pad=0.15,ticks=np.linspace(np.nanmin(U),np.nanmax(U),4))
        # In the second window, plot the meridionally averaged streamfunction
        x = np.linspace(0,360,50)
        xz,zx = np.meshgrid(x,q['z_d'][1:-1]/1000,indexing='ij')
        psi = np.outer(np.cos(2*x*np.pi/180),xt[:n]) - np.outer(np.sin(2*x*np.pi/180),xt[n:2*n])
        psi *= np.exp(q['z_d'][1:-1]/(2*q['H']))*3/8*q['length']**2/q['time']
        im = ax[1].contourf(xz,zx,psi,cmap='seismic')
        fig.colorbar(im,ax=ax[1],format=ticker.FuncFormatter(fmt),
                orientation='horizontal',pad=0.15,ticks=np.linspace(np.nanmin(psi),np.nanmax(psi),4))
        ax[1].set_xlabel("Longitude",fontdict=font)
        ax[1].set_ylabel("Height (km)",fontdict=font)
        ax[1].set_title("Streamfunction {}".format(suffix),fontdict=font)
        return fig,ax
    def plot_two_snapshots(self,xt0,xt1,suffix0="",suffix1=""):
        q = self.q
        n = q['Nz']-1
        def fmt(num,pos):
            return '{:.1f}'.format(num)
        fig,ax = plt.subplots(ncols=2,figsize=(12,6))
        z = q['z_d'][1:-1]/1000
        handles = []
        handle, = ax[0].plot(xt0[2*n:3*n]*q['length']/q['time'],z,color='skyblue',linewidth=1.5,label=suffix0)
        handles += [handle]
        handle, = ax[0].plot(xt1[2*n:3*n]*q['length']/q['time'],z,color='red',linewidth=1.5,label=suffix1)
        handles += [handle]
        ax[0].set_xlabel(r"$U$ (m/s)",fontdict=ffont)
        ax[0].set_ylabel("Altitude (km)",fontdict=ffont)
        ax[0].set_title("Zonal wind",fontdict=ffont)
        ax[0].tick_params(axis='both', which='major', labelsize=20)
        base_x = 20.0
        base_y = 20.0
        xlim,ylim = ax[0].get_xlim(),ax[0].get_ylim()
        ax[0].xaxis.set_major_locator(plt.FixedLocator(np.arange(xlim[0]//base_x,xlim[-1]//base_x+1,1)*base_x))
        ax[0].yaxis.set_major_locator(plt.FixedLocator(np.arange(ylim[0]//base_y,ylim[-1]//base_y+1,1)*base_y))
        if len(suffix0)>0 or len(suffix1)>0:
            ax[0].legend(handles=handles,prop={'size': 25})
        # In the second window, plot the meridionally averaged streamfunction
        x = np.linspace(0,360,50)
        xz,zx = np.meshgrid(x,q['z_d'][1:-1]/1000,indexing='ij')
        psi0 = np.outer(np.cos(2*x*np.pi/180),xt0[:n]) - np.outer(np.sin(2*x*np.pi/180),xt0[n:2*n])
        psi1 = np.outer(np.cos(2*x*np.pi/180),xt1[:n]) - np.outer(np.sin(2*x*np.pi/180),xt1[n:2*n])
        psi1 *= np.exp(q['z_d'][1:-1]/(2*q['H']))*q['length']**2/q['time']
        psi0 *= np.exp(q['z_d'][1:-1]/(2*q['H']))*q['length']**2/q['time']
        im = ax[1].contour(xz,zx,psi0,colors='skyblue')
        im = ax[1].contour(xz,zx,psi1,colors='red')
        ax[1].set_xlabel("Longitude",fontdict=ffont)
        #ax[1].set_ylabel("Altitude (km)",fontdict=font)
        dpsi = im.levels[1]-im.levels[0]
        ax[1].set_title(r"$\Psi$ (m$^2$/s) ($\Delta=%.1e$)"%(dpsi),fontdict=ffont)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
        base_x = 90.0
        base_y = 20.0
        xlim,ylim = ax[1].get_xlim(),ax[1].get_ylim()
        ax[1].xaxis.set_major_locator(plt.FixedLocator(np.arange(xlim[0]//base_x,xlim[-1]//base_x+1,1)*base_x))
        ax[1].yaxis.set_major_locator(plt.NullLocator()) #plt.FixedLocator(np.arange(ylim[0]//base_y,ylim[-1]//base_y+1,1)*base_y))
        return fig,ax
    def plot_sparse_regression_zslices(self,coeffs,scores,savefolder,suffix=""):
        # Plot the correlation as a function of altitude, where the regression has been done for each altitude separately
        funlib = self.observable_function_library()
        z = self.q['z_d'][1:-1]/1000
        theta_names = self.regression_feature_names() #[r"$Re(\Psi)$",r"$Im(\Psi)$",r"$U$",funlib["mag"]["name"],funlib["vT"]["name"],funlib["dqdy"]["name"]][0:]
        #theta_names = [r"$Re(\Psi)$",r"$Im(\Psi)$",r"$U$"]
        method = 'LASSO'
        fig,ax = plt.subplots(ncols=2,figsize=(12,6),sharey=True,constrained_layout=True)
        handles = []
        for i in range(len(theta_names)):
            handle, = ax[0].plot(coeffs[:,i],z,label=theta_names[i])
            handles += [handle]
        ax[0].set_xlabel("Coefficient",fontdict=font)
        ax[0].set_ylabel("Altitude (km)",fontdict=font)
        ax[0].set_title(r"{} Coefficients$(z)$".format(method),fontdict=font)
        ax[0].tick_params(axis='both', which='major', labelsize=20)
        ax[0].legend(handles=handles, prop={'size': 15}, loc='upper right')
        coeffrange = np.ptp(coeffs)
        ax[0].set_xlim([np.min(coeffs)-0.1*coeffrange,np.max(coeffs)+0.4*coeffrange])
        ax[1].plot(scores,z,color='black')
        ax[1].set_xlabel(r"$R^2$",fontdict=font)
        ax[1].set_title("{} Correlation$(z)$".format(method),fontdict=font)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
        fig.savefig(join(savefolder,"lasso_coeffs_zdep_{}_{}".format(method,suffix)))
        plt.close(fig)
        return
    def plot_sparse_regression_allz(self,beta,score,savefolder,suffix=""):
        z = self.q['z_d'][1:-1]/1000
        n = self.q['Nz']-1
        #theta_names = [r"$Re(\Psi)$",r"$Re(\Psi_z)$",r"$Im(\Psi)$",r"$Im(\Psi_z)$",r"$U$",r"$U_z$"]
        funlib = self.observable_function_library()
        theta_names = self.regression_feature_names() #[r"$Re(\Psi)$",r"$Im(\Psi)$",r"$U$",funlib["mag"]["name"],funlib["vT"]["name"],funlib["dqdy"]["name"]][0:]
        Nth = len(theta_names)
        print("Nth = {}".format(Nth))
        fig,ax = plt.subplots(ncols=Nth,figsize=(2*Nth,6),sharey=True,constrained_layout=True)
        for i in range(Nth):
            ax[i].barh(z,beta[i*n:(i+1)*n],color='black')
            ax[i].set_xlim([-np.max(np.abs(beta)),np.max(np.abs(beta))])
            ax[i].plot(np.zeros(2),z[[0,-1]],linestyle='-',color='black')
            ax[i].locator_params(tight=True,nbins=2,axis='x')
            ax[i].xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax[i].tick_params(axis='both', which='major', labelsize=14)
            ax[i].set_title(theta_names[i],fontdict={'family':'serif', 'size': 20})
        ax[0].set_ylabel(r"$z\,(\mathrm{km})$",fontdict=font)
        fig.suptitle("LASSO Coefficients; $R^2=%.2f$" % (score),fontsize=25)
        fig.savefig(join(savefolder,"lasso"+suffix))
        plt.close(fig)
        print("R2 = {}".format(score))
        reg_coeffs = np.concatenate((z.reshape((n,1)),beta.reshape((Nth,n)).T),axis=1)
        np.savetxt(join(savefolder,"reg_coeffs{}.txt".format(suffix)), reg_coeffs, fmt='%.3e')
        print("savefolder = {}".format(savefolder))
        return
    def fun_at_level(self,fun,z):
        q = self.q
        # For any function returning a z-dependent function, restrict it to a level
        def funz(x):
            i = np.argmin(np.abs(q['z_d'][1:-1]/1000 - z))
            return fun(x)[:,i:i+1]
        return funz
    def fun_zmean(self,fun):
        def funz(x):
            return np.mean(fun(x),1).reshape((len(x),1))
        return funz
    def plot_multiple_states(self,X,qlevels,qsymbol,colorlist=None):
    #def plot_zdep_family_weighted(self,cv_x,cv_a,cv_b,labels,weights=None,cv_name=None,colorlist=None,units=1.0,unit_symbol=""):
        # Given a sequence of states (X) plot them all on the same graph. For the Holton-Mass model, this means different zonal wind profiles.
        key = "U"
        funlib = self.observable_function_library()
        U = funlib[key]["fun"](X)
        units = funlib[key]["units"]
        Ua,Ub = funlib[key]["fun"](self.xst)
        num_states = len(X) 
        if len(X) != len(qlevels): sys.exit("Need same number of transition states as qlevels")
        q = self.q
        fig,ax = plt.subplots(figsize=(6,6))
        # Plot a bunch of zonal wind profiles
        #N = len(cv_x)
        z = q['z_d'][1:-1]/1000
        ax.plot(units*Ua,z,color='blue',linestyle='--',linewidth=4)
        ax.plot(units*Ub,z,color='red',linestyle='--',linewidth=4)
        if colorlist is None: colorlist = plt.cm.coolwarm(qlevels)
        handles = []
        for i in range(num_states):
            handle, = ax.plot(units*U[i],z,alpha=1.0,color=colorlist[i],linewidth=3,label=r"$%s=%.2f$"%(qsymbol,qlevels[i]))
            if i % len(np.unique(qlevels)) == 0: handles += [handle]
        #ax.legend(handles=handles,loc='lower right',prop={'size':18})
        ax.set_ylabel(r"$z\,(km)$",fontdict=font)
        ax.set_xlabel("{} ({})".format(funlib[key]['name'],funlib[key]['unit_symbol']),fontdict=font)
        #if cv_name is not None:
        #    xlab = cv_name
        #    if len(unit_symbol) > 0: xlab += " (%s)"%(unit_symbol)
        #    ax.set_xlabel(xlab,fontdict=font)
        return fig,ax

def unit_test():
    # Calculate the drift of a random vector and compare it to that from the hm_funs file
    np.random.seed(0)
    x0 = np.random.randn(1000,75)
    # First the old method
    qold = hm.default_physical_params()
    qold = hm.initialize_physical_params(qold)
    time_old = time.perf_counter()
    for i in range(100):
        xdot_old = hm.time_derivative_batch(x0,qold)
    time_old = time.perf_counter() - time_old
    # Now, the new method
    qnew = default_parameters()
    qnew = initialize_params(qnew)
    drift_fun = get_drift_fun(qnew)
    time_new = time.perf_counter()
    for i in range(100):
        xdot_new = drift_fun(x0,qnew)
    time_new = time.perf_counter() - time_new
    print("Max abs (xdot_new - xdot_old) = \n{}".format(np.max(np.abs(xdot_new - xdot_old))))
    print("Old time = {}".format(time_old))
    print("New time = {}".format(time_new))
    # 2. Find the fixed points
    n = qnew['Nz']-1
    diffusion_fun = get_diffusion_fun(qnew)
    model = Model(3*n,drift_fun,diffusion_fun,0.005)
    model.find_fixed_points(approximate_fixed_points(qnew),tmax=500)
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))
    ax[0].plot(model.xst[0,:n],qnew['z_d'][1:-1]/1000,color='blue')
    ax[0].plot(model.xst[1,:n],qnew['z_d'][1:-1]/1000,color='red')
    ax[0].set_xlabel("Re(Psi)")
    ax[0].set_ylabel("Alitude (km)")
    ax[1].plot(model.xst[0,n:2*n],qnew['z_d'][1:-1]/1000,color='blue')
    ax[1].plot(model.xst[1,n:2*n],qnew['z_d'][1:-1]/1000,color='red')
    ax[1].set_xlabel("Im(Psi)")
    ax[1].set_ylabel("Alitude (km)")
    ax[2].plot(model.xst[0,2*n:3*n],qnew['z_d'][1:-1]/1000,color='blue')
    ax[2].plot(model.xst[1,2*n:3*n],qnew['z_d'][1:-1]/1000,color='red')
    ax[2].set_xlabel("U")
    ax[2].set_ylabel("Alitude (km)")
    fig.savefig(os.path.join(savefolder,"fixed_points"))
    # 3. Plot a timeseries
    t = np.linspace(0,5000,10000)
    x0 = model.xst[:1]
    x = model.integrate_euler_maruyama(x0,print_interval=50)
    fig,ax = plt.subplots()
    ax.plot(t,x[:,0,2*n+qnew['zi']])
    ax.set_xlabel("Time")
    ax.set_ylabel("U({} km)".format(ref_alt))
    fig.savefig(os.path.join(savefolder,"timeseries"))
    return

def first_derivative(F,lower,upper,dz):
    # Compute the first z derivatives any field
    Nt,n = F.shape
    Fz = np.zeros([Nt,n])
    Fz[:,1:-1] = (F[:,2:n] - F[:,0:n-2]) / (2*dz)
    Fz[:,0] = (F[:,1] - lower)/(2*dz)
    Fz[:,-1] = (upper - F[:,-2])/(2*dz)
    return Fz

def D1mat(n,dz,lower_value,lower_type,upper_value,upper_type):
    # coeff is the coefficient to multiply the derivative by
    D = np.zeros((n,n))
    C = np.zeros(n)
    D[np.arange(1,n-1),np.arange(2,n)] = 1.0/(2*dz)
    D[np.arange(1,n-1),np.arange(n-2)] = -1.0/(2*dz)
    # Lower boundary
    if lower_type == 'dirichlet':
        D[0,1] = 1.0/(2*dz)
        C[0] = -lower_value/(2*dz)
    elif lower_type == 'neumann':
        D[0,[0,1]] = 2.0/3*np.array([-1,1])/dz
        C[0] = 1.0/3*lower_value
    # Upper boundary
    if upper_type == 'dirichlet':
        D[n-1,n-2] = -1.0/(2*dz)
        C[n-1] = upper_value/(2*dz)
    elif upper_type == 'neumann':
        D[n-1,[n-2,n-1]] = 2.0/3*np.array([-1,1])/dz
        C[n-1] = 1.0/3*upper_value
    return C,D

def D2mat(n,dz,lower_value,lower_type,upper_value,upper_type):
    D = np.zeros((n,n))
    C = np.zeros(n)
    D[np.arange(1,n-1),np.arange(1,n-1)] = -2.0/dz**2
    D[np.arange(1,n-1),np.arange(n-2)] = 1.0/dz**2
    D[np.arange(1,n-1),np.arange(2,n)] = 1.0/dz**2
    # Lower boundary
    if lower_type == 'dirichlet':
        D[0,[0,1]] = np.array([-2,1])/dz**2
        C[0] = lower_value/dz**2
    elif lower_type == 'neumann':
        D[0,[0,1]] = np.array([-1,1])/(1.5*dz**2)
        C[0] = -1.0/(1.5*dz)*lower_value
    # Upper boundary
    if upper_type == 'dirichlet':
        D[n-1,[n-2,n-1]] = np.array([1,-2])/dz**2
        C[n-1] = upper_value/dz**2
    elif upper_type == 'neumann':
        D[n-1,[n-2,n-1]] = -np.array([-1,1])/(1.5*dz**2)
        C[n-1] = 1.0/(1.5*dz)*upper_value
    return C,D

def second_derivative(F,lower,upper,dz):
    # Compute the first z derivatives any field
    Nt,n = F.shape
    Fzz = np.zeros([Nt,n])
    Fzz[:,1:-1] = (F[:,2:n] + F[:,0:n-2] - 2*F[:,1:-1]) / (dz**2)
    Fzz[:,0] = (F[:,1] + lower - 2*F[:,0])/(dz**2)
    Fzz[:,-1] = (upper + F[:,-2] - 2*F[:,-1])/(dz**2)
    return Fzz
if __name__ == "__main__":
    unit_test()
