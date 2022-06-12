
# This is where the Holton-Mass model is specified
import numpy as np
from numpy import load,save
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'monospace', 'size': 10,}
font = {'family': 'monospace', 'size': 18,}
ffont = {'family': 'monospace', 'size': 25}
bigfont = {'family': 'monospace', 'size': 40}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mpl_colors
import scipy.sparse as sps
from scipy.interpolate import interp1d
import time
import os
from os.path import join,exists
import sys
from model_obj import Model
import helper

asymb = r"$\mathbf{a}$"
bsymb = r"$\mathbf{b}$"

class HoltonMassModel(Model):
    def __init__(self,physical_params,xst=None):
        self.ref_alt = physical_params['ref_alt'] # in kilometers
        q = {
           'rad': 6370.0e3, 'day': 24*3600.0, 'g': 9.82, 'phi0': np.pi/3, 
           'sx': 2, 'sy': 3, 'zB_d': 0.0, 'zT_d': 70.0e3, 'H': 7.0e3, 
           'Omega': 2*np.pi/(24*3600), 'Nsq_d': 4.0e-4, 'ideal_gas_constant': 8.314,
           'eps': 8.0/(3*np.pi), 'UR_0_d': 10.0, 'gamma': 1.5, 'hB_d': physical_params['hB_d'], 
           'nfreq': 3, 'Nz': 26, 'length': 2.5e5, 'time': 24*3600.0,
           'du_per_day': physical_params['du_per_day'], 'dt_sim': physical_params['dt_sim'],
        }
        # Corrections
        q['ideal_gas_constant'] = 287.0 # Joules / (kilogram * Kelvin) (for dry air)
        self.q = self.initialize_params(q)
        # E[(dU)^2]/dt = (du_perday m/s)^2/day
        # E[(dpsi)^2]/dt = (g/f0*dh_perday m^2/s)^2/day
        self.abdefdim = physical_params['abdefdim']
        self.radius_a = physical_params['radius_a']
        self.radius_b = physical_params['radius_b']
        self.state_dim = 3*(self.q['Nz']-1)
        tpt_obs_dim = self.state_dim # We're dealing with full state here
        self.noise_rank = self.q['nfreq']
        self.dt_sim = self.q['dt_sim']
        parallel_sim_limit = 10000
        nshort_per_file_limit = 100000
        super().__init__(3*(self.q['Nz']-1),q['nfreq'],q['dt_sim'],tpt_obs_dim,parallel_sim_limit,nshort_per_file_limit)
        x0_list = self.approximate_fixed_points()
        if xst is None:
            self.find_fixed_points(tmax=600)
            print(f"self.xst.shape = {self.xst.shape}")
        else:
            self.xst = xst
            print(f"self.xst.shape = {self.xst.shape}")
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
        rhs[:,2*n:3*n] = ((q['alpha_z'][1:-1]-q['alpha'][1:-1])*q['UR_z'][1:-1]) + q['alpha'][1:-1]*q['UR_zz'][1:-1] 
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
        radius_a = self.radius_a #8.0
        if self.abdefdim == 75:
            da = np.sqrt(np.sum((cvx - cva)**2, 1))
        else:
            n = self.q['Nz']-1
            zi = self.q['zi']
            da = cva[2*n+zi] - cvx[:,2*n+zi]
        return np.maximum(0, da-radius_a)
    def bdist(self,cvx):
        cvb = self.tpt_observables(self.xst[1])
        radius_b = self.radius_b # 30.0 
        n = self.q['Nz']-1
        zi = self.q['zi']
        if self.abdefdim == 75:
            db = np.sqrt(np.sum((cvx - cvb)**2, 1))
        elif self.abdefdim == 2: # Defined in (U, IHF) space
            u_x,u_b = cvx[:,2*n+zi],cvb[2*n+zi]
            ihf_x = self.integrated_meridional_heat_flux(cvx)[:,zi]
            ihf_b = self.integrated_meridional_heat_flux(np.array([cvb]))[0,zi]
            db = np.sqrt((u_x-u_b)**2 + (ihf_x-ihf_b)**2)
        else:
            db = cvx[:,2*n+zi] - cvb[2*n+zi]
        return np.maximum(0, db-radius_b)
    def set_param_folder(self):
        self.param_foldername = ("du{}_h{}".format(self.q['du_per_day'],self.q['hB_d'])).replace(".","p")
        return
    def plot_least_action_scalars(self,physical_param_folder,obs_names=["Uref","magref"],fig=None,ax=None,negtime=False):
        # Plot scalars evolving over time
        funlib = self.observable_function_library()
        # Given the noise forcing, plot a picture of the least action pathway
        # Also add a vertical line for when U(30 km) first goes negative
        q = self.q
        n = q['Nz']-1
        sig = q['sig_mat']
        z = q['z_d'][1:-1]/1000
        # -------------------- A -> B ----------------------
        wmin = load(join(physical_param_folder,"wmin_dirn1.npy"))
        xmin = load(join(physical_param_folder,"xmin_dirn1.npy"))
        tmin = load(join(physical_param_folder,"tmin_dirn1.npy"))
        # Bound the times by entering A and B
        adist = self.adist(xmin)
        bdist = self.bdist(xmin)
        tmin_idx0 = np.where((adist>0)*(bdist>0))[0][0]
        if np.min(bdist) <= 0:
            tmin_idx1 = np.where(bdist<=0)[0][0]
        else:
            tmin_idx1 = np.where((adist>0)*(bdist>0))[0][-1]
        tmin = tmin[tmin_idx0:tmin_idx1+1]
        wmin = wmin[tmin_idx0:tmin_idx1+1]
        xmin = xmin[tmin_idx0:tmin_idx1+1]
        if negtime: tmin -= tmin[-1]
        if fig is None or ax is None: # Must be a nx1 array of ax, where n=len(ob_names)
            fig,ax = plt.subplots(nrows=len(obs_names),ncols=1,figsize=(6,6*len(obs_names)),sharex=True)
        # Find where U(30 km) drops below b
        uref_xst = funlib["Uref"]["fun"](self.tpt_obs_xst)
        #uref_xst[0] -= self.radius_a
        #uref_xst[1] += self.radius_b
        #ulb_idx = np.where(funlib["Uref"]["fun"](self.tpt_observables(xmin)) < uref_xst[1] + self.radius_b)[0][0]
        #print("ulb_idx = {}, tmin[ulb_idx] = {}".format(ulb_idx,tmin[ulb_idx]))
        for i in range(len(obs_names)):
            #axi = ax if len(obs_names)==1 else ax[i]
            # Top row: U(30 km) 
            obs_xst = funlib[obs_names[i]]["fun"](self.tpt_obs_xst)
            obs = funlib[obs_names[i]]["fun"](self.tpt_observables(xmin))
            units = funlib[obs_names[i]]["units"]
            unit_symbol = funlib[obs_names[i]]["unit_symbol"]
            ax[i].plot(tmin,units*obs,color='black')
            ax[i].plot(tmin[[0,-1]],units*(obs_xst[0]-0*self.radius_a)*np.ones(2),color='skyblue',linewidth=3)
            ax[i].plot(tmin[[0,-1]],units*(obs_xst[1]+0*self.radius_b)*np.ones(2),color='red',linewidth=3)
            #ax[i].plot(np.mean(tmin[ulb_idx:ulb_idx+2])*np.ones(2),units*np.array([np.min(obs),np.max(obs)]),color='black',linestyle='--')
            #ax[i].axvline(x=np.mean(tmin[ulb_idx:ulb_idx+2]),color='black',linestyle='--')
            ax[i].set_ylabel("%s [%s]"%(funlib[obs_names[i]]["name"],funlib[obs_names[i]]["unit_symbol"]),fontdict=font)
            ax[i].set_xlabel(r"Time to $B$ [days]",fontdict=font)
            ax[i].set_title(r"Minimum-action path ($A\to B$)",fontdict=font)
        return fig,ax
    def plot_least_action_profiles(self,physical_param_folder,prof_names=["U","mag"],fig=None,ax=None,negtime=False,logscale=False):
        funlib = self.observable_function_library()
        # Given the noise forcing, plot a picture of the least action pathway
        q = self.q
        n = q['Nz']-1
        sig = q['sig_mat']
        z = q['z_d'][1:-1]/1000
        # -------------------- A -> B ----------------------
        wmin = load(join(physical_param_folder,"wmin_dirn1.npy"))
        xmin = load(join(physical_param_folder,"xmin_dirn1.npy"))
        tmin = load(join(physical_param_folder,"tmin_dirn1.npy"))
        # Bound the times by entering A and B
        adist = self.adist(xmin)
        bdist = self.bdist(xmin)
        tmin_idx0 = np.where((adist>0)*(bdist>0))[0][0]
        if np.min(bdist) <= 0:
            tmin_idx1 = np.where(bdist<=0)[0][0]
        else:
            tmin_idx1 = np.where((adist>0)*(bdist>0))[0][-1]
        tmin = tmin[tmin_idx0:tmin_idx1+1]
        wmin = wmin[tmin_idx0:tmin_idx1+1]
        xmin = xmin[tmin_idx0:tmin_idx1+1]
        if negtime: tmin -= tmin[-1]
        tz,zt = np.meshgrid(tmin,z,indexing='ij')
        uref_xst = funlib["Uref"]["fun"](self.tpt_obs_xst)
        #ulb_idx = np.where(funlib["Uref"]["fun"](self.tpt_observables(xmin)) < uref_xst[1])[0][0]
        #dU = (sig.dot(wmin.T)).T[:,2*n:3*n] # This part is specific to U
        if fig is None or ax is None: # Must be a (n+1)x1 array of ax, where n=len(prof_names)
            fig,ax = plt.subplots(nrows=len(prof_names),ncols=1,figsize=(6,18),sharex=True)
        ims = []
        for i in range(len(prof_names)):
            name = funlib[prof_names[i]]["name"]
            obs = funlib[prof_names[i]]["fun"](self.tpt_observables(xmin))
            units = funlib[prof_names[i]]["units"]
            unit_symbol = funlib[prof_names[i]]["unit_symbol"]
            eps = np.min(np.abs(obs))/2
            if logscale:
                im = ax[i].contourf(tz,zt,np.maximum(eps,obs)*units,cmap=plt.cm.coolwarm,norm=mpl_colors.LogNorm(vmin=eps,vmax=np.max(np.abs(obs))))
            else:
                im = ax[i].contourf(tz,zt,units*obs,cmap=plt.cm.coolwarm)
            #ax[i].axvline(x=np.mean(tmin[ulb_idx:ulb_idx+2]),color='black',linestyle='--')
            ims += [im]
            fig.colorbar(im,ax=ax[i])
            ax[i].set_ylabel(r"$z$ [km]",fontdict=font)
            ax[i].set_title(r"Minimum-action %s$(z)$ profile [%s]"%(name,unit_symbol),fontdict=font)
            ax[i].set_xlabel(r"$-\eta_B^+$ [days]",fontdict=font)
        # Save
        #fig.savefig(join(savefolder,"fw_ab_plot"))
        #plt.close(fig)
        # B -> A
        #wmin = load(join(physical_param_folder,"wmin_dirn-1.npy"))
        #xmin = load(join(physical_param_folder,"xmin_dirn-1.npy"))
        #tmin = load(join(physical_param_folder,"tmin_dirn-1.npy"))
        #obs = funlib[fun_name]["fun"](self.tpt_observables(xmin))
        #dU = (sig.dot(wmin.T)).T[:,2*n:3*n]
        #z = q['z_d'][1:-1]/1000
        #tz,zt = np.meshgrid(tmin,z,indexing='ij')
        #ax[0,1].plot(tmin,units*obs[:,q['zi']],color='black')
        #ax[0,1].plot(tmin[[0,-1]],units*obs_xst[0,q['zi']]*np.ones(2),color='skyblue')
        #ax[0,1].plot(tmin[[0,-1]],units*obs_xst[1,q['zi']]*np.ones(2),color='red')
        #im = ax[1,1].contourf(tz,zt,units*obs,cmap=plt.cm.coolwarm)
        #im = ax[2,1].contourf(tz[:-1,:],zt[:-1,:],dU,cmap=plt.cm.coolwarm)
        ## Common legends
        ##fig.suptitle("Least action paths",fontdict=bigfont)
        #ax[0,0].set_title(r"$A\to B$ least action",fontdict=bigfont)
        #ax[0,1].set_title(r"$B\to A$ least action",fontdict=bigfont)
        #ax[1,0].set_title("%s"%(funlib[fun_name]["name"]),fontdict=bigfont)
        #ax[1,1].set_title("%s"%(funlib[fun_name]["name"]),fontdict=bigfont)
        #ax[2,0].set_title(r"$\delta U(t)$",fontdict=bigfont)
        #ax[2,1].set_title(r"$\delta U(t)$",fontdict=bigfont)
        #ax[0,0].set_ylabel("%s(%d km) (%s)"%(funlib[fun_name]["name"],self.ref_alt,funlib[fun_name]["unit_symbol"]),fontdict=bigfont)
        #ax[1,0].set_ylabel(r"$z\,(\mathrm{km})$",fontdict=bigfont)
        #ax[2,0].set_ylabel(r"$z\,(\mathrm{km})$",fontdict=bigfont)
        ## Tick labels
        #for i in range(ax.shape[0]):
        #    for j in range(ax.shape[1]):
        #        ax[i,j].tick_params(axis='both',labelsize=30)
        ## Save
        #fig.savefig(join(physical_param_folder,"fw_plot_%s"%fun_name))
        #plt.close(fig)
        return fig,ax,ims
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
                'vTintref_l0': {
                    'pay': lambda x: 1*(funlib['vTintref']['fun'](x)*funlib['vTintref']['units'] < 0),
                    'name': r"$1\{$%s$ < 0\ \mathrm{K}\cdot\mathrm{m}^2/\mathrm{s}\}$"%funlib["vTintref"]["name"],
                    },
                'one': {
                    'pay': lambda x: np.ones(len(x)),
                    'name': '1',
                    },
                'heatflux': {
                    'pay': funlib['vTref']['fun'], 
                    'name': funlib['vTref']['name'],
                    },
                'heatflux_g60': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 60.0),
                    'name':  r"$1\{$%s$ > 60\ \mathrm{K}\cdot\mathrm{m/s}\}$"%funlib['vTref']['name'],
                    },
                'heatflux_g1em5': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 1e-5),
                    'name':  r"$1\{$%s$ > 1\times10^{-5}\ \mathrm{K}\cdot\mathrm{m/s}\}$"%funlib['vTref']['name'],
                    },
                'heatflux_g1p5em5': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 1.5e-5),
                    'name':  r"$1\{$%s$ > 1.5\times10^{-5}\ \mathrm{K}\cdot\mathrm{m/s}\}$"%funlib['vTref']['name'],
                    },
                'heatflux_g2em5': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 2e-5),
                    'name':  r"$1\{$%s$ > 2\times10^{-5}\ \mathrm{K}\cdot\mathrm{m/s}\}$"%funlib['vTref']['name'],
                    },
                'heatflux_g2p5em5': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 2.5e-5),
                    'name':  r"$1\{$%s$ > 2.5\times10^{-5}\ \mathrm{K}\cdot\mathrm{m/s}\}$"%funlib['vTref']['name'],
                    },
                'heatflux_g3em5': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 3e-5),
                    'name':  r"$1\{$%s$ > 3\times10^{-5}\ \mathrm{K}\cdot\mathrm{m/s}\}$"%funlib['vTref']['name'],
                    },
                'heatflux_g5em5': {
                    'pay': lambda x: 1*(funlib['vTref']['fun'](x)*funlib['vTref']['units'] > 5e-5),
                    'name':  r"$1\{$%s$ > 5\times10^{-5}\ \mathrm{K}\cdot\mathrm{m/s}\}$"%funlib['vTref']['name'],
                    },
                'magref': {
                    'pay': funlib['magref']['fun'],
                    'name': funlib['magref']['name'],
                    },
                'magref_g6e6': {
                    'pay': lambda x: 1.0*(funlib['magref']['fun'](x)*funlib['magref']['units'] > 6e6),
                    'name': r"$1\{$%s$ > 6\times 10^6\ \mathrm{m}^2/\mathrm{s}\}$"%funlib['magref']['name'],
                    },
                'magref_g8e6': {
                    'pay': lambda x: 1.0*(funlib['magref']['fun'](x)*funlib['magref']['units'] > 8e6),
                    'name': r"$1\{$%s$ > 8\times 10^6\ \mathrm{m}^2/\mathrm{s}\}$"%funlib['magref']['name'],
                    },
                'magref_g1e7': {
                    'pay': lambda x: 1.0*(funlib['magref']['fun'](x)*funlib['magref']['units'] > 1e7),
                    'name': r"$1\{$%s$ > 10^7\ \mathrm{m}^2/\mathrm{s}\}$"%funlib['magref']['name'],
                    },
                'Uref': {
                    'pay': funlib["Uref"]["fun"],
                    'name': funlib['Uref']['name'],
                    },
                'Uref_l0': {
                    'pay': lambda x: 1.0*(funlib['Uref']['fun'](x) < 0),
                    'name': r"$1\{$%s$ < 0\ \mathrm{m/s}\}$"%funlib['Uref']['name'],
                    },
                'Uref_ln10': {
                    'pay': lambda x: 1.0*(funlib['Uref']['fun'](x)*funlib['Uref']['units'] < -10),
                    'name':  r"$1\{$%s$ < -10\ \mathrm{m/s}\}$"%funlib['Uref']['name'],
                    },
                'Uref_ln15': {
                    'pay': lambda x: 1.0*(funlib['Uref']['fun'](x)*funlib['Uref']['units'] < -15),
                    'name':  r"$1\{$%s$ < -15\ \mathrm{m/s}\}$"%funlib['Uref']['name'],
                    },
                'Uref_ln20': {
                    'pay': lambda x: 1.0*(funlib['Uref']['fun'](x)*funlib['Uref']['units'] < -20),
                    'name':  r"$1\{$%s$ < -20\ \mathrm{m/s}\}$"%funlib['Uref']['name'],
                    },
                }
        self.dam_dict = {
                'one': {
                    'pay': lambda x: np.ones(len(x)),
                    'name': 'Time',
                    'name_fwd': "\\tau^+", #r"$\tau^+$",
                    'name_bwd': "-\\tau^-", #r"$\tau^-$",
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
                'heatflux': {
                    'pay': self.fun_at_level(funlib["vT"]["fun"], self.ref_alt),
                    'name': 'Heat flux (%.0f km)'%self.ref_alt,
                    'name_fwd': "\\int_0^{\\tau^+}\\overline{v'T'}(%.0f\\ \\mathrm{km})dt"%self.ref_alt, 
                    'name_bwd': "\\int_{\\tau^-}^0\\overline{v'T'}(%.0f\\ \\mathrm{km})dt"%self.ref_alt, #r"$\tau^-$",
                    'name_full': "\\int_{\\tau^-}^{\\tau^+}\\overline{v'T'}(%.0f\\ \\mathrm{km})\,dt"%self.ref_alt, #r"$\tau^+-\tau^-$",
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
        if k == 0:
            Xder = X #*np.exp(q['z'][1:-1]/2)
        else:
            Xz = first_derivative(X,lower,upper,q['dz'])
            if k == 1:
                Xder = (Xz + 0.5*X) #*np.exp(q['z'][1:-1]/2)
            elif k == 2:
                Xzz = second_derivative(X,lower,upper,q['dz'])
                Xder = (Xzz + Xz + 0.25*X) #*np.exp(q['z'][1:-1]/2)
            else:
                raise Exception(f"You asked for the {k}th derivative, but I only compute them up to k=2")
        return Xder*np.exp(q['z'][1:-1]/2)
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
    def eddy_enstrophy_projected(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        X,Y = x[:,:n],x[:,n:2*n]
        Xz = first_derivative(X,q['Psi0'],0,q['dz'])
        Yz = first_derivative(Y,0,0,q['dz'])
        Xzz = second_derivative(X,q['Psi0'],0,q['dz'])
        Yzz = second_derivative(Y,0,0,q['dz'])
        delta = q['Gsq']*(q['k']**2+q['l']**2) + 1.0/4
        Qsq = delta**2*(X**2 + Y**2)
        Qsq += Xzz**2 + Yzz**2
        Qsq -= 2*delta*(X*Xzz + Y*Yzz)
        Qsq *= 0.5*np.exp(q['z'][1:-1])
        return Qsq
    def eddy_enstrophy(self,x,lat=60):
        # This is corrected to deal with the squaredness. 
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        # Compute all the necessary correlations.
        X,Y = x[:,:n],x[:,n:2*n]
        X0,Y0 = self.product_rule_z(X,q['Psi0'],0,0),self.product_rule_z(Y,0,0,0)
        X1,Y1 = self.product_rule_z(X,q['Psi0'],0,1),self.product_rule_z(Y,0,0,1)
        X2,Y2 = self.product_rule_z(X,q['Psi0'],0,2),self.product_rule_z(Y,0,0,2)
        # term 1: (k^2+l^2)^2*(psi')^2
        enstrophy = -(q['k']**2 + q['l']**2)**2 * (X0**2 + Y0**2)/2
        # term 2: (1/G^4) * (psi'_zz)^2
        enstrophy += 1/q['Gsq']**2 * (X2**2 + Y2**2)/2
        # term 3: (1/G^4) * (1/H^2) * (psi'_z)^2
        enstrophy += 1/q['Gsq']**2 * (X1**2 + Y1**2)/2
        # term 4: (1/G^4) * (-2/H) * (psi'_zz)*(psi'_z)
        enstrophy -= 2/q['Gsq']**2 * (X1*X2 + Y1*Y2)/2
        # term 5: -2/(G^2) * (k^2+l^2) + (psi'*psi'_zz)
        enstrophy -= 2/q['Gsq'] * (q['k']**2 + q['l']**2) * (X0*X2 + Y0*Y2)/2 
        # term 6: 2/(G^2) * (k^2+l^2) * 1/H * (psi'*psi'_z)
        enstrophy += 2/q['Gsq'] * (q['k']**2 + q['l']**2) * (X0*X1 + Y0*Y1)/2
        # Fix the latitude
        enstrophy *= 1/2*np.sin(q['sy']*lat*np.pi/180)**2
        # Old mistaken code:
        #qpsq = (q['k']**2 + q['l']**2)*(X**2 + Y**2)
        #qpsq += -1/q['Gsq']*(q['k']**2 + q['l']**2) * (
        #        X*X2 + Y*Y2 - X*X1 - Y*Y1)
        #qpsq += 1/q['Gsq']**2 * (X2**2 + Y2**2 - 2*(X1*X2 + Y1*Y2) + X1**2 + Y1**2)
        #qpsq *= 0.5*17/35
        return enstrophy
    def wave_activity_projected(self,x):
        Qsq = self.eddy_enstrophy_projected(x)
        dqdy = self.background_pv_gradient_projected(x)
        eps = 1e-10
        wa = Qsq / (dqdy + 1*(np.abs(dqdy) < eps))
        wa[np.abs(dqdy)<eps] = np.nan
        return wa
    def meridional_heat_flux(self,x):
        q = self.q
        # Compute the meridional heat flux, perhaps of the whole timeseries
        n = q['Nz']-1
        Nt = len(x)
        heat_flux = np.ones([Nt,q['Nz']-1])
        heat_flux *= q['k'] 
        #heat_flux *= q['k_d']*q['H']*q['f0_d']/(2*q['ideal_gas_constant'])
        heat_flux *= np.exp(q['z'][1:-1])
        # Now it has to be multiplied by vertical derivatives
        Xz = first_derivative(x[:,:n],q['Psi0'],0,q['dz']) 
        Yz = first_derivative(x[:,n:2*n],0,0,q['dz'])
        heat_flux *= (x[:,:n]*Yz - x[:,n:2*n]*Xz)
        return heat_flux
    def weighted_meridional_heat_flux(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        heat_flux = np.ones([Nt,n])
        heat_flux *= q['k'] #q['k_d']*q['H']*q['f0_d']/(2*q['ideal_gas_constant'])
        #heat_flux *= np.exp(q['z'][:-1])*17.0/35
        # Now it has to be multiplied by vertical derivatives
        Xz = first_derivative(x[:,:n],q['Psi0'],0,q['dz']) 
        Yz = first_derivative(x[:,n:2*n],0,0,q['dz'])
        Yz0 = (4*x[:,2*n] - x[:,2*n+1])/(2*q['dz'])
        heat_flux *= (x[:,:n]*Yz - x[:,n:2*n]*Xz)
        heat_flux *= np.exp(-q['z'][1:-1])
        return heat_flux
    def unweighted_vertical_average(self,fz):
        # for a field f(z), integrate it vertically with no weighting
        q = self.q
        Nx,n = fz.shape
        if n != q['Nz'] - 1:
            raise Exception(f"You gave me a field of shape {fz.shape}, but dim 1 must have size {n}")
        weight = np.ones(n) #np.exp(-q['z'][1:-1])
        F = np.sum(fz*weight, axis=1) / np.sum(weight)
        return F
    def weighted_vertical_average(self,fz):
        # for a field f(z), integrate it vertically weighted by density. 
        q = self.q
        Nx,n = fz.shape
        if n != q['Nz'] - 1:
            raise Exception(f"You gave me a field of shape {fz.shape}, but dim 1 must have size {n}")
        weight = np.exp(-q['z'][1:-1])
        F = np.sum(fz*weight, axis=1) / np.sum(weight)
        return F
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
    def dqdy_relaxation_projected(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        U = x[:,2*n:]
        U_upper = 1.0/3*(4*U[:,-1] - U[:,-2] + 2*q['dz']*q['UR_z'][-1])
        Uz = first_derivative(U,q['UR_0'],U_upper,q['dz'])
        Uzz = second_derivative(U,q['UR_0'],U_upper,q['dz'])
        a,az = q['alpha'][1:-1],q['alpha_z'][1:-1]
        relax = (az-a)*(Uz - q['UR_z'][1:-1]) + a*(Uzz - q['UR_zz'][1:-1])
        print(f"relax.shape = {relax.shape}")
        return relax
    def background_pv_gradient_projected(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        U = x[:,2*n:]
        U_upper = 1.0/3*(4*U[:,-1] - U[:,-2] + 2*q['dz']*q['UR_z'][-1])
        Uz = first_derivative(U,q['UR_0'],U_upper,q['dz'])
        Uzz = second_derivative(U,q['UR_0'],U_upper,q['dz'])
        pvgrad = q['Gsq']*(q['beta'] + q['eps']*q['l']**2*U)
        pvgrad += q['eps']*(Uz - Uzz)
        return pvgrad
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
    def meridional_pv_flux_projected(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        X,Y = x[:,:n],x[:,n:2*n]
        Xz = first_derivative(X,q['Psi0'],0,q['dz'])
        Yz = first_derivative(Y,0,0,q['dz'])
        Xzz = second_derivative(X,q['Psi0'],0,q['dz'])
        Yzz = second_derivative(Y,0,0,q['dz'])
        pvflux = q['k']*(X*Yzz - Y*Xzz)
        pvflux *= np.exp(q['z'][1:-1])
        return pvflux
    def meridional_pv_flux(self,x,lat=60):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        X,Y = x[:,:n],x[:,n:2*n]
        X0,Y0 = self.product_rule_z(X,q['Psi0'],0,0),self.product_rule_z(Y,0,0,0)
        X1,Y1 = self.product_rule_z(X,q['Psi0'],0,1),self.product_rule_z(Y,0,0,1)
        X2,Y2 = self.product_rule_z(X,q['Psi0'],0,2),self.product_rule_z(Y,0,0,2)
        pv_flux = q['k']/q['Gsq']*(X0*Y2-Y0*X2)/2 
        pv_flux += q['k']/q['Gsq']*(-X0*Y1+Y0*X1)/2
        pv_flux *= np.sin(q['sy']*lat*np.pi/180)**2
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
    def test_pvgrad_equation(self,x,dt=None):
        q = self.q
        Nx = len(x)
        n = q['Nz']-1
        if dt is None: dt = self.dt_sim
        xdot = self.drift_fun(x)
        funlib = self.observable_function_library()
        relax = funlib["dqdyrelax"]["fun"](x) #*funlib["dqdyrelax"]["units"]
        vq = funlib["vq"]["fun"](x) #*funlib["vq"]["units"]
        dqdy_dot = (funlib["dqdy"]["fun"](x + dt*xdot) - funlib["dqdy"]["fun"](x - dt*xdot))/(2*dt) #* q["time"] #* funlib["dqdy"]["units"]
        net = 2/(q['eps']*q['l'])**2 * dqdy_dot - 2/(q['eps']*q['l']**2) * relax - vq
        return dqdy_dot,relax,vq,net
    def test_gramps(self,x,dt=None):
        q = self.q
        Nx = len(x)
        n = q['Nz']-1
        if dt is None: dt = self.dt_sim
        xdot = self.drift_fun(x)
        funlib = self.observable_function_library()
        relax = funlib["dqdyrelax"]["fun"](x) 
        diss = funlib["diss"]["fun"](x)
        dqdy = funlib["dqdy"]["fun"](x)
        enst_dot = (funlib["enstrophy"]["fun"](x + dt*xdot) - funlib["enstrophy"]["fun"](x - dt*xdot))/(2*dt)
        gramps_dot = (funlib["gramps"]["fun"](x + dt*xdot) - funlib["gramps"]["fun"](x - dt*xdot))/(2*dt)
        net = enst_dot + gramps_dot - 2/(q['eps']*q['l']**2)*relax*dqdy - diss
        return enst_dot,gramps_dot,relax,dqdy,diss,net
    def test_enstrophy_equation(self,x,lat=60,dt=None):
        # Compute the left-hand side of Eq. 2-11 of Yoden 1987 to check if it's zero. 
        # The time derivative of enstrophy will have to be approximated as a finite difference. 
        q = self.q
        Nx = len(x)
        n = q['Nz']-1
        if dt is None: dt = self.dt_sim
        xdot = self.drift_fun(x)
        units_flag = True 
        if units_flag:
            funlib = self.observable_function_library()
            pvflux = funlib["vq"]["fun"](x)*funlib["vq"]["units"]
            pvgrad = funlib["dqdy"]["fun"](x)*funlib["dqdy"]["units"]
            diss = funlib["diss"]["fun"](x)*funlib["diss"]["units"]
            q2_0 = funlib["enstrophy"]["fun"](x)*funlib["enstrophy"]["units"]
            q2_1 = funlib["enstrophy"]["fun"](x + dt*xdot)*funlib["enstrophy"]["units"]
            enstrophy_tendency = (q2_1 - q2_0)/(dt*q["time"])
        else:
            pvflux = self.meridional_pv_flux_projected(x)
            pvgrad = self.background_pv_gradient_projected(x)
            diss = self.dissipation_projected(x)
            q2_0 = self.eddy_enstrophy_projected(x)
            q2_1 = self.eddy_enstrophy_projected(x + xdot*dt)
            enstrophy_tendency = (q2_1 - q2_0)/dt
        # Now compute the terms with finite difference
        lhs = enstrophy_tendency + pvflux*pvgrad - diss
        print(f"dt = {dt}, Max abs LHS = {np.max(np.abs(lhs))}")
        print(f"enstrophy_tendency mean = {np.mean(np.abs(enstrophy_tendency),axis=1)}")
        print(f"pvflux mean = {np.mean(np.abs(pvflux),axis=1)}")
        print(f"pvgrad mean = {np.mean(np.abs(pvgrad),axis=1)}")
        print(f"diss mean = {np.mean(np.abs(diss),axis=1)}")
        return enstrophy_tendency,pvflux,pvgrad,diss,lhs
    def dissipation_projected(self,x):
        q = self.q
        n = q['Nz']-1
        Nt = len(x)
        X,Y = x[:,:n],x[:,n:2*n]
        Xz = first_derivative(X,q['Psi0'],0,q['dz'])
        Xzz = second_derivative(X,q['Psi0'],0,q['dz'])
        Yz = first_derivative(Y,0,0,q['dz'])
        Yzz = second_derivative(Y,0,0,q['dz'])
        a = q['alpha'][1:-1]
        az = q['alpha_z'][1:-1]
        delta = q['Gsq']*(q['k']**2+q['l']**2) + 1.0/4
        diss = delta*(az/2 - a/4)*(X**2 + Y**2)
        diss += delta*az*(X*Xz + Y*Yz)
        diss += (a*(1.0/4 + delta) - az/2)*(Xzz*X + Yzz*Y)
        diss -= az*(Xzz*Xz + Yzz*Yz)
        diss -= a*(Xzz**2 + Yzz**2)
        #diss = -delta*(
        #        (a/4-az/2)*(X**2 + Y**2)
        #        -az*(X*Xz + Y*Yz)
        #        -a*(X*Xzz + Y*Yzz)
        #        )
        #diss += (a/4-az/2)*(X*Xzz + Y*Yzz)
        #diss -= az*(Xz*Xzz + Yz*Yzz)
        #diss -= a*(Xzz**2 + Yzz**2)
        diss *= np.exp(q['z'][1:-1])
        return diss
    def dissipation_old(self,x,lat=60):
        q = self.q
        # The dissipation term from Eq. 2-11 of Yoden 1987 (but not dividided by dq/dy), and with a minus sign. 
        n = q['Nz']-1
        Nt = len(x)
        X,Y = x[:,:n],x[:,n:2*n]
        X0,Y0 = self.product_rule_z(X,q['Psi0'],0,0),self.product_rule_z(Y,0,0,0)
        X1,Y1 = self.product_rule_z(X,q['Psi0'],0,1),self.product_rule_z(Y,0,0,1)
        X2,Y2 = self.product_rule_z(X,q['Psi0'],0,2),self.product_rule_z(Y,0,0,2)
        a = q['alpha'][1:-1]
        az = q['alpha_z'][1:-1]
        diss = 1/q['Gsq'] * (q['k']**2+q['l']**2)*(
            (az-a)*(X0*X1 + Y0*Y1)/2 + a*(X0*X2 + Y0*Y2)/2)
        diss += 1/q['Gsq']**2 * (
            - (az-2*a)*(X1*X2 + Y1*Y2)/2 
            + (az-a)*(X1**2 + Y1**2)/2 
            - a*(X2**2 + Y2**2)/2)
        diss *= np.sin(q['sy']*lat*np.pi/180)**2
        # Old code:
        #diss = 1/q['Gsq']*((q['k']**2+q['l']**2)*(X*X2 + Y*Y2)/2
        #        + (az - a)*(X*X1 + Y*Y1)/2)
        #diss += 1/q['Gsq']**2 * (-a*(X2**2 + Y2**2)/2
        #        + (a - az)*(X2*X1 + Y2*Y1)/2)
        return diss
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
                "U": {
                    "fun": lambda X: X[:,2*n:3*n],
                    "name":r"$U$",
                    "units": q['length']/q['time'],
                    "unit_symbol": r"m/s",
                    "name_english": "Zonal wind",
                    "abbrv": "U",
                 },
                "Uref":
                {"fun": lambda X: X[:,2*n+q['zi']],
                 "name":r"$U$(%.0f km)"%self.ref_alt,
                 "units": q['length']/q['time'],
                 "unit_symbol": r"m/s",
                 "abbrv": "U30",
                 },
                "U21p5":
                {"fun": lambda X: X[:,2*n+zlevel(21.5)],
                 "name":r"$U(21.5\,\mathrm{km})$",
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
                 "name":r"$U(67\,\mathrm{km})$",
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
                "vTint": {
                        "fun": lambda X: self.integrated_meridional_heat_flux(X),
                        "name": r"IHF$(z)$", #"$\int_{0}^{z}e^{-z/H}\overline{v'T'}dz$",
                        "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                        "unit_symbol": r"K$\cdot$m$^2/$s",
                        "name_english": "Integrated heat flux",
                        "abbrv": "VTI",
                 },
                "vTint13p5": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,zlevel(13.5)],
                 "name": r"IHF(13.5 km)", #r"$\int_{0\ km}^{%.1f\ km}e^{-z/H}\overline{v'T'}dz$"%(13.5),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "vTint19": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,zlevel(19)],
                 "name": r"IHF(19 km)", #$\int_{0\ km}^{%.1f\ km}e^{-z/H}\overline{v'T'}dz$"%(19),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "vTint21p5": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,zlevel(21.5)],
                 "name": r"IHF(21.5 km)", #$\int_{0\ km}^{%.1f\ km}e^{-z/H}\overline{v'T'}dz$"%(21.5),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"K$\cdot$m$^2$/s",
                 },
                "vTintref": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,q['zi']],
                 "name": r"IHF(30 km)", #$\int_{0\ \mathrm{km}}^{%.0f\ \mathrm{km}}e^{-z/H}\overline{v'T'}dz$"%(self.ref_alt),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"K$\cdot$m$^2/$s",
                 },
                "vTinttop": 
                {"fun": lambda X: self.integrated_meridional_heat_flux(X)[:,-1],
                 "name": r"IHF(%.1f km)"%(q['z_d'][-2]/1000), #$\int_{z_b}^{%.0f}e^{-z/H}\overline{v'T'}dz$"%(q['z_d'][-2]/1000),
                 "units": q['H']**2*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"$K\cdot m^2/s$",
                 },
                "wvT": # ADJUSTED UNITS 
                {"fun": lambda X: self.weighted_meridional_heat_flux(X),
                 "name": r"$e^{-z/H}\overline{v'T'}$",
                 "units": q['H']*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"K$\cdot$m/s",
                 },
                "vT": # ADJUSTED UNITS 
                {"fun": lambda X: self.meridional_heat_flux(X),
                 "name": r"$\overline{v'T'}$",
                 "units": q['H']*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"K$\cdot$m/s",
                 "name_english": "Heat flux",
                 "abbrv": "VT",
                 },
                "vT21p5": 
                {"fun": lambda X: self.meridional_heat_flux(X)[:,zlevel(21.5)],
                 "name": r"$\overline{v'T'}(21.5\,\mathrm{km})$", 
                 "units": q['H']*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"K$\cdot\mathrm{m/s}$",
                 },
                "vT67": 
                {"fun": lambda X: self.meridional_heat_flux(X)[:,zlevel(67)],
                 "name": r"$\overline{v'T'}(67\,\mathrm{km})$", #$\int_{0\ km}^{%.1f\ km}e^{-z/H}\overline{v'T'}dz$"%(21.5),
                 "units": q['H']*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"K$\cdot$m/s",
                 },
                "vTref": 
                {"fun": lambda X: self.meridional_heat_flux(X)[:,q['zi']],
                 "name": r"$\overline{v'T'}(%.0f\,\mathrm{km})$"%self.ref_alt,
                 "units": q['H']*q['f0_d']/(2*q['length']*q['ideal_gas_constant'])*q['length']**4/(q['H']*q['time']**2),
                 "unit_symbol": r"$\mathrm{K}\cdot\mathrm{m/s}$",
                 },
                "dqdy": {
                        "fun": lambda X: self.background_pv_gradient_projected(X),
                        "name": r"$\beta_e$", #r"$\partial_y\overline{q}$",
                        "units": 1/q["Gsq"]*1/(q['length']*q['time']),
                        "unit_symbol": "m$^{-1}$s$^{-1}$",
                        "name_english": "Mean PV gradient",
                        "abbrv": "Be",
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
                 "unit_symbol": r"m$^{-1}$s$^{-1}$",
                 },
                "epflux_z":
                {"fun": lambda X: self.epflux_z(X),
                 "name":r"$e^{-z/H}\overline{v'q'}$",
                 "units": q['length']**3/(q['H']**2*q['time']**2),
                 "unit_symbol": r"s$^{-1}$",
                 },
                "vq": {
                        "fun": lambda X: self.meridional_pv_flux_projected(X),
                        "name": r"$F_q$", #r"$\overline{v'q'}$",
                        "units": 1/q["Gsq"]*q["length"]/q["time"]**2,
                        "unit_symbol": r"m s$^{-2}$",
                        "name_english": "Eddy PV flux",
                        "abbrv": "VQ",
                        },
                "enstrophy": {
                        "fun": lambda X: self.eddy_enstrophy_projected(X),
                        "name": r"$\mathcal{E}$", #$\frac{1}{2}\overline{q'^2}$",
                        "units": 1/q["Gsq"]**2*1/q['time']**2,
                        "unit_symbol": r"s$^{-2}$",
                        "name_english": "Eddy enstrophy",
                        "abbrv": "E",
                        },
                "diss": {
                        "fun": lambda X: self.dissipation_projected(X),
                        "name": r"$D$", #r"$\frac{f_0^2}{N^2}\overline{q'\rho_s^{-1}\partial_z(\alpha\rho_s\partial_z\psi')}$",
                        "units": 1/q["Gsq"]**2*1/q["time"]**3,
                        "unit_symbol": "s$^{-3}$",
                        "name_english": "Eddy enstrophy dissipation",
                        "abbrv": "D",
                 },
                "ensttend": {
                        "fun": lambda X: self.dissipation_projected(X) - self.meridional_pv_flux_projected(X)*self.background_pv_gradient_projected(X),
                        "name": r"$\partial_t\mathcal{E}$ inferred",
                        "units": 1/q["Gsq"]**2*1/q["time"]**3,
                        "unit_symbol": r"s$^{-3}$",
                        "name_english": r"Inferred enstrophy tendency",
                        "abbrv": "ET",
                        },
                "dqdyrelax": {
                        "fun": lambda X: self.dqdy_relaxation_projected(X),
                        "name": r"$\frac{\epsilon\ell^2}{2}R$", #r"$\rho^{-1}\partial_z[\rho\alpha(U - U^R)_z]$",
                        "units": 1/q["Gsq"]*1/(q['length']*q['time']**2),
                        "unit_symbol": r"m$^{-1}$s$^{-2}$",
                        "name_english": r"Relaxation of PV gradient",
                        "abbrv": "RofBe",
                        },

            }
        funs["diss_ref"] = {
                "fun": lambda X: funs["diss"]["fun"](X)[:,q['zi']],
                "name": r"%s (30 km)"%(funs["diss"]["name"]),
                "units": funs["diss"]["units"],
                "unit_symbol": funs["diss"]["unit_symbol"],
                "name_english": funs["diss"]["name_english"],
                "abbrv": "D30",
                }
        funs["diss_int"] = {
                "fun": lambda X: self.unweighted_vertical_average(funs["diss"]["fun"](X)),
                "name": r"$\overline{D}$",
                "units": funs["diss"]["units"],
                "unit_symbol": funs["diss"]["unit_symbol"],
                "name_english": r"Integrated %s"%(funs["diss"]["name_english"]),
                "abbrv": "DI",
                }
        funs["dqdy_times_vq"] = {
                "fun": lambda X: funs["dqdy"]["fun"](X)*funs["vq"]["fun"](X),
                "name": r"%s%s"%(funs["dqdy"]["name"],funs["vq"]["name"]),
                "units": funs["dqdy"]["units"]*funs["vq"]["units"],
                "unit_symbol": "s$^{-3}$",
                "name_english": "Meridional PV advection",
                "abbrv": "BeVQ",
               }
        funs["dqdy_times_vq_ref"] = {
                "fun": lambda X: funs["dqdy"]["fun"](X)[:,q['zi']]*funs["vq"]["fun"](X)[:,q['zi']],
                "name": r"%s%s (30 km)"%(funs["dqdy"]["name"],funs["vq"]["name"]),
                "units": funs["dqdy"]["units"]*funs["vq"]["units"],
                "unit_symbol": "s$^{-3}$",
                "name_english": "Meridional PV advection at 30 km",
                "abbrv": "BeVQ30",
               }
        funs["dqdy_times_vq_int"] = {
                "fun": lambda X: self.unweighted_vertical_average(funs["dqdy"]["fun"](X)*funs["vq"]["fun"](X)),
                "name": r"$\overline{F_q\beta_e}$",
                "units": funs["dqdy"]["units"]*funs["vq"]["units"],
                "unit_symbol": "s$^{-3}$",
                "name_english": "$z$-averaged Meridional PV advection",
                "abbrv": "BeVQI",
               }
        funs["waveactproj"] = {
                "fun": lambda X: self.wave_activity_projected(X),
                "name": r"$\overline{q'^2}/(2\partial_y\overline{q})$",
                "units": funs["enstrophy"]["units"]/funs["dqdy"]["units"],
                "unit_symbol": "m/s",
                "name_english": "Wave activity",
                "abbrv": "W",
                }
        funs["gramps"] = {
                "fun": lambda X: funs["dqdy"]["fun"](X)**2/(q['eps']*q['l'])**2, 
                "name": r"$\Gamma$",
                "units": 1/q["Gsq"]**2 * 1/q["time"]**2, 
                "unit_symbol": r"s$^{-2}$",
                "name_english": "GRAMPS",
                "abbrv": "G",
                }
        funs["gramps_relax_coeff"] = {
                "fun": lambda X: funs["dqdyrelax"]["fun"](X)*2/(q['eps']*q['l']**2),
                "name": r"R",
                "units": q["length"]**2*funs["dqdyrelax"]["units"],
                "unit_symbol": "s$^{-3}$",
                "name_english": "GRAMPS relaxation coefficient",
                "abbrv": "R",
                }
        funs["gramps_relax"] = {
                "fun": lambda X: funs["gramps_relax_coeff"]["fun"](X)*funs["dqdy"]["fun"](X), #funs["dqdyrelax"]["fun"](X)*funs["dqdy"]["fun"](X)*2/(q['eps']*q['l']**2),
                "name": r"%s%s"%(funs["gramps_relax_coeff"]["name"],funs["dqdy"]["name"]),
                "units": q["length"]**2*funs["dqdyrelax"]["units"]*funs["dqdy"]["units"],
                "unit_symbol": "s$^{-3}$",
                "name_english": "GRAMPS relaxation",
                "abbrv": "RBe",
                }
        funs["gramps_relax_ref"] = {
                "fun": lambda X: funs["gramps_relax"]["fun"](X)[:,q['zi']], 
                "name": r"%s (30 km)"%(funs["gramps_relax"]["name"]),
                "units": funs["gramps_relax"]["units"],
                "unit_symbol": funs["gramps_relax"]["unit_symbol"],
                "name_english": r"%s (30 km)"%(funs["gramps_relax"]["name_english"]),
                "abbrv": "RBe30",
                }
        funs["gramps_relax_int"] = {
                "fun": lambda X: self.unweighted_vertical_average(funs["gramps_relax"]["fun"](X)),
                "name": r"$\overline{R\beta_e}$",
                "units": funs["gramps_relax"]["units"],
                "unit_symbol": funs["gramps_relax"]["unit_symbol"],
                "name_english": r"$z$-averaged %s"%(funs["gramps_relax"]["name_english"]),
                "abbrv": "RBeI",
                }
        funs["gramps_plus_enstrophy"] = {
                "fun": lambda X: funs["gramps"]["fun"](X) + funs["enstrophy"]["fun"](X),
                "name": "%s$+$%s"%(funs["gramps"]["name"],funs["enstrophy"]["name"]),
                "units": funs["enstrophy"]["units"],
                "unit_symbol": "s$^{-2}$",
                "name_english": "%s + %s"%(funs["gramps"]["name_english"],funs["enstrophy"]["name_english"]),
                "abbrv": "G+E",
                }
        funs["gramps_plus_enstrophy_ref"] = {
                "fun": lambda X: (funs["gramps"]["fun"](X)[:,q['zi']] + funs["enstrophy"]["fun"](X)[:,q['zi']]),
                "name": "%s$+$%s (30 km)"%(funs["gramps"]["name"],funs["enstrophy"]["name"]),
                "units": funs["enstrophy"]["units"],
                "unit_symbol": "s$^{-2}$",
                "name_english": "Enstrophy + GRAMPS at $z=30$ km",
                "abbrv": "G+E30",
                }
        funs["gramps_plus_enstrophy_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["gramps"]["fun"](X) + funs["enstrophy"]["fun"](X)),
                "name": r"(%s$+$%s)$^{1/2}$"%(funs["gramps"]["name"],funs["enstrophy"]["name"]),
                "units": np.sqrt(funs["enstrophy"]["units"]),
                "unit_symbol": "s$^{-1}$",
                "name_english": "(%s + %s)$^{1/2}$"%(funs["gramps"]["name_english"],funs["enstrophy"]["name_english"]),
                "abbrv": "G+Esqrt",
                }
        funs["gramps_plus_enstrophy_sqrt_ref"] = {
                "fun": lambda X: funs["gramps_plus_enstrophy_sqrt"]["fun"](X)[:,q['zi']],
                "name": r"%s (30 km)"%(funs["gramps_plus_enstrophy_sqrt"]["name"]),
                "units": funs["gramps_plus_enstrophy_sqrt"]["units"],
                "unit_symbol": funs["gramps_plus_enstrophy_sqrt"]["unit_symbol"],
                "name_english": r"%s (30 km)"%(funs["gramps_plus_enstrophy_sqrt"]["name_english"]),
                "abbrv": "G+E30sqrt"
                }
        # --------- GRAMPS stuff -------------
        funs["gramps_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["gramps"]["fun"](X)),
                "name": r"%s$^{1/2}$"%(funs["gramps"]["name"]),
                "units": np.sqrt(funs["gramps"]["units"]),
                "unit_symbol": r"s$^{-1}$",
                "name_english": "%s$^{1/2}$"%(funs["gramps"]["name_english"]),
                "abbrv": "Gsqrt",
                }
        funs["gramps_ref"] = {
                "fun": lambda X: funs["gramps"]["fun"](X)[:,q['zi']],
                "name": "%s (30 km)"%(funs["gramps"]["name"]),
                "units": funs["gramps"]["units"],
                "unit_symbol": funs["gramps"]["unit_symbol"],
                "name_english": "%s ($z=30$km)"%(funs["gramps"]["name_english"]),
                "abbrv": "G30",
                }
        funs["gramps_ref_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["gramps"]["fun"](X)[:,q['zi']]),
                "name": "%s (30 km)"%(funs["gramps_sqrt"]["name"]),
                "units": np.sqrt(funs["gramps"]["units"]),
                "unit_symbol": funs["gramps_sqrt"]["unit_symbol"],
                "name_english": "%s ($z=30$km)"%(funs["gramps_sqrt"]["name_english"]),
                "abbrv": "G30sqrt",
                }
        funs["gramps_int"] = {
                "fun": lambda X: self.unweighted_vertical_average(funs["gramps"]["fun"](X)),
                "name": r"$\overline{\Gamma}$",
                "units": funs["gramps"]["units"],
                "unit_symbol": "%s"%(funs["gramps"]["unit_symbol"]),
                "name_english": "GRAMPS ($z$-mean)",
                "abbrv": "GI",
                }
        funs["gramps_int_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["gramps_int"]["fun"](X)),
                "name": "(%s)$^{1/2}$"%(funs["gramps_int"]["name"]),
                "units": np.sqrt(funs["gramps_int"]["units"]),
                "unit_symbol": "s$^{-1}$",
                "name_english": "(%s)$^{1/2}$"%(funs["gramps_int"]["name_english"]),
                "abbrv": "GIsqrt",
                }
        # --------- Enstrophy stuff -------------
        funs["enstrophy_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["enstrophy"]["fun"](X)),
                "name": r"%s$^{1/2}$"%(funs["enstrophy"]["name"]),
                "units": np.sqrt(funs["enstrophy"]["units"]),
                "unit_symbol": r"s$^{-1}$",
                "name_english": "Enstrophy$^{1/2}$",
                "abbrv": "Esqrt",
                }
        funs["enstrophy_ref"] = {
                "fun": lambda X: funs["enstrophy"]["fun"](X)[:,q['zi']],
                "name": "%s (30 km)"%(funs["enstrophy"]["name"]),
                "units": funs["enstrophy"]["units"],
                "unit_symbol": funs["enstrophy"]["unit_symbol"],
                "name_english": "%s ($z=30$km)"%(funs["enstrophy"]["name_english"]),
                "abbrv": "E30",
                }
        funs["enstrophy_ref_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["enstrophy"]["fun"](X)[:,q['zi']]),
                "name": "%s (30 km)"%(funs["enstrophy_sqrt"]["name"]),
                "units": funs["enstrophy_sqrt"]["units"],
                "unit_symbol": r"s$^{-1}$",
                "name_english": "%s$^{1/2}$ ($z=30$km)"%(funs["enstrophy"]["name_english"]),
                "abbrv": "E30sqrt",
                }
        funs["enstrophy_int"] = {
                "fun": lambda X: self.unweighted_vertical_average(funs["enstrophy"]["fun"](X)),
                "name": r"$\overline{\mathcal{E}}$",
                "units": funs["enstrophy"]["units"],
                "unit_symbol": funs["enstrophy"]["unit_symbol"],
                "name_english": "Enstrophy ($z$-mean)",
                "abbrv": "EI",
                }
        funs["enstrophy_int_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["enstrophy_int"]["fun"](X)),
                "name": "(%s)$^{1/2}$"%(funs["enstrophy_int"]["name"]),
                "units": np.sqrt(funs["enstrophy_int"]["units"]),
                "unit_symbol": "s$^{-1}$",
                "name_english": "(%s)$^{1/2}$"%(funs["enstrophy_int"]["name_english"]),
                "abbrv": "EIsqrt",
                }
        # ---------- Action-angle stuff -----------
        ang_const = 1.0
        funs["gramps_enstrophy_angle"] = {
                "fun": lambda X: np.arctan2(ang_const*funs["enstrophy_sqrt"]["fun"](X), funs["gramps_sqrt"]["fun"](X)),
                "name": r"$\tan^{-1}$(%i%s/%s)"%(ang_const,funs["gramps_sqrt"]["name"],funs["enstrophy_sqrt"]["name"]),
                "units": 1.0, 
                "unit_symbol": "radians",
                "name_english": "Enstrophy-GRAMPS angle",
                "abbrv": "GEANG",
                }
        funs["enstrophy_gramps_ratio"] = {
                "fun": lambda X: funs["enstrophy_sqrt"]["fun"](X)/funs["gramps_sqrt"]["fun"](X),
                "name": "%s/%s"%(funs["enstrophy_sqrt"]["name"],funs["gramps_sqrt"]["name"]),
                "units": 1.0, 
                "unit_symbol": "",
                "name_english": "Enstrophy/GRAMPS",
                "abbrv": "EoG",
                }
        funs["enstrophy_gramps_ratio_ref"] = {
                "fun": lambda X: funs["enstrophy_gramps_ratio"]["fun"](X)[:,q['zi']],
                "name": r"%s (30 km)"%(funs["enstrophy_gramps_ratio"]["name"]),
                "units": funs["enstrophy_gramps_ratio"]["units"],
                "unit_symbol": funs["enstrophy_gramps_ratio"]["unit_symbol"],
                "name_english": r"%s (30 km)"%(funs["enstrophy_gramps_ratio"]),
                "abbrv": "%s30"%(funs["enstrophy_gramps_ratio"]["abbrv"]),
                }
        funs["gramps_enstrophy_arclength"] = {
                "fun": lambda X: np.arctan2(ang_const*funs["enstrophy_sqrt"]["fun"](X), funs["gramps_sqrt"]["fun"](X))*funs["gramps_plus_enstrophy_sqrt"]["fun"](X),
                "name": r"Arc(%s,%s)"%(funs["gramps"]["name"],funs["enstrophy"]["name"]),
                "units": funs["gramps_plus_enstrophy_sqrt"]["units"],
                "unit_symbol": funs["gramps_plus_enstrophy_sqrt"]["unit_symbol"],
                "name_english": "Enstrophy-GRAMPS arclength",
                "abbrv": "GEARC",
                }
        funs["gramps_enstrophy_arclength_ref"] = {
                "fun": lambda X: funs["gramps_enstrophy_arclength"]["fun"](X)[:,q['zi']],
                "name": r"%s (30 km)"%(funs["gramps_enstrophy_arclength"]["name"]),
                "units": funs["gramps_enstrophy_arclength"]["units"],
                "unit_symbol": funs["gramps_enstrophy_arclength"]["unit_symbol"],
                "name_english": r"%s (30 km)"%(funs["gramps_enstrophy_arclength"]),
                "abbrv": "GEARC30",
                }
        # Vertical integrals of nonlinear action-angle coordinates
        funs["gramps_plus_enstrophy_sqrt_int"] = {
                "fun": lambda X: self.unweighted_vertical_average(funs["gramps_plus_enstrophy_sqrt"]["fun"](X)),
                "name": r"$\overline{(\Gamma+\mathcal{E})^{1/2}}$",
                "units": np.sqrt(funs["gramps_plus_enstrophy"]["units"]),
                "unit_symbol": r"s$^{-1}$",
                "name_english": r"(%s)$^{1/2}$ ($z$-mean)"%(funs["gramps_plus_enstrophy"]["name_english"]),
                "abbrv": "GEsqrtI",
                }
        funs["gramps_enstrophy_arclength_int"] = {
                "fun": lambda X: self.unweighted_vertical_average(funs["gramps_enstrophy_arclength"]["fun"](X)),
                "name": r"$\overline{\mathrm{Arc}(\Gamma,\mathcal{E})}$",
                "units": funs["gramps_enstrophy_arclength"]["units"],
                "unit_symbol": funs["gramps_enstrophy_arclength"]["unit_symbol"],
                "name_english": r"%s ($z$-mean)"%(funs["gramps_enstrophy_arclength"]["name_english"]),
                "abbrv": "GEARCI",
                }
        # Action-angle coordinates of vertical integrals
        funs["gramps_plus_enstrophy_int"] = {
                "fun": lambda X: funs["gramps_int"]["fun"](X) + funs["enstrophy_int"]["fun"](X),
                "name": r"$\overline{\Gamma}+\overline{\mathcal{E}}$",
                "units": funs["gramps_int"]["units"],
                "unit_symbol": r"s$^{-2}$",
                "name_english": r"GRAMPS + Enstrophy ($z$-mean)",
                "abbrv": "GEI",
                }
        funs["gramps_plus_enstrophy_int_sqrt"] = {
                "fun": lambda X: np.sqrt(funs["gramps_int"]["fun"](X) + funs["enstrophy_int"]["fun"](X)),
                "name": r"$(\overline{\Gamma}+\overline{\mathcal{E}})^{1/2}$",
                "units": np.sqrt(funs["gramps_int"]["units"]),
                "unit_symbol": r"s$^{-1}$",
                "name_english": r"(GRAMPS + Enstrophy $z$-mean)$^{1/2}$",
                "abbrv": "GEIsqrt",
                }
        funs["gramps_enstrophy_int_arclength"] = {
                "fun": lambda X: funs["gramps_plus_enstrophy_int_sqrt"]["fun"](X) * np.arctan2(funs["enstrophy_int_sqrt"]["fun"](X), funs["gramps_int_sqrt"]["fun"](X)),
                "units": funs["gramps_plus_enstrophy_int_sqrt"]["units"],
                "unit_symbol": funs["gramps_plus_enstrophy_int_sqrt"]["unit_symbol"],
                "name": r"Arc($\overline{\Gamma},\overline{\mathcal{E}}$)",
                "name_english": "Arclength of integral",
                "abbrv": "GEIARC",
                }
        funs["gramps_enstrophy_angle_ref"] = {
                "fun": lambda X: funs["gramps_enstrophy_angle"]["fun"](X)[:,q['zi']], 
                "name": r"%s (30 km)"%(funs["gramps_enstrophy_angle"]["name"]),
                "units": 1.0, 
                "unit_symbol": "radians",
                "name_english": "Enstrophy-GRAMPS angle (30 km)",
                "abbrv": "GEANG30",
                }
        return funs
    def plot_cooling_profile(self):
        q = self.q
        fig,ax = plt.subplots()
        ax.plot(q['alpha_d'],q['z_d']/1000,color='black')
        ax.set_xlabel(r"$\alpha(z)$ [s$^{-1}$]", fontdict=font)
        ax.set_ylabel(r"$z$ [km]", fontdict=font)
        ax.set_title(r"Newtonian cooling")
        ax.axvline(x=0, linestyle='--', color='black')
        return fig,ax
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
        ax[0].set_ylabel("Height [km]",fontdict=bigfont)
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
        ax[1].set_ylabel("Height [km]",fontdict=font)
        ax[1].set_title("Streamfunction {}".format(suffix),fontdict=font)
        return fig,ax
    def plot_two_snapshots(self,xt0,xt1,suffix0="",suffix1=""):
        q = self.q
        n = q['Nz']-1
        def fmt(num,pos):
            return '{:.1f}'.format(num)
        fig,ax = plt.subplots(ncols=3,figsize=(18,6),sharey=True)
        z = q['z_d'][1:-1]/1000
        handles = []
        handle, = ax[0].plot(xt0[2*n:3*n]*q['length']/q['time'],z,color='skyblue',linewidth=1.5,label=suffix0)
        handles += [handle]
        handle, = ax[0].plot(xt1[2*n:3*n]*q['length']/q['time'],z,color='red',linewidth=1.5,label=suffix1)
        handles += [handle]
        ax[0].set_xlabel(r"$U$ [m/s]",fontdict=ffont)
        ax[0].set_ylabel(r"$z$ [km]",fontdict=ffont)
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
        #ax[1].set_ylabel("Altitude [km]",fontdict=font)
        dpsi = im.levels[1]-im.levels[0]
        ax[1].set_title(r"Streamfunction [m$^2$/s]",fontdict=ffont) # ($\Delta=%.1e$)"%(dpsi),fontdict=ffont)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
        base_x = 90.0
        base_y = 20.0
        xlim,ylim = ax[1].get_xlim(),ax[1].get_ylim()
        #ax[1].xaxis.set_major_locator(plt.FixedLocator(np.arange(xlim[0]//base_x,xlim[-1]//base_x+1,1)*base_x))
        #ax[1].yaxis.set_major_locator(plt.NullLocator()) #plt.FixedLocator(np.arange(ylim[0]//base_y,ylim[-1]//base_y+1,1)*base_y))
        ax[2].plot(q['alpha_d'][1:-1], z, color='black')
        ax[2].set_xlabel(r"$\alpha(z)$ [s$^{-1}$]",fontdict=ffont)
        ax[2].set_title("Cooling coefficient",fontdict=ffont)
        ax[2].axvline(x=0, linestyle='--', color='black')
        ax[2].tick_params(axis='both', which='major', labelsize=20)
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
        ax[0].set_ylabel("Altitude [km]",fontdict=font)
        ax[0].set_title(r"{} Coefficients$(z)$".format(method),fontdict=font)
        ax[0].tick_params(axis='both', which='major', labelsize=20)
        ax[0].legend(handles=handles, prop={'size': 15}, loc='upper right')
        coeffrange = np.ptp(coeffs)
        ax[0].set_xlim([np.min(coeffs)-0.1*coeffrange,np.max(coeffs)+0.4*coeffrange])
        ax[1].plot(scores,z,color='black')
        ax[1].set_xlabel(r"$R^2$",fontdict=font)
        ax[1].set_title("{} Correlation$(z)$".format(method),fontdict=font)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
        fig.savefig(join(savefolder,"lasso_coeffs_zdep_{}{}".format(method,suffix)))
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
        np.savetxt(join(savefolder,"lasso_coeffs{}.txt".format(suffix)), reg_coeffs, fmt='%.3e')
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
    def plot_state_distribution_signed(self,X,rflux,rflux_idx,qlevels,qsymbol,colors=None,key="U",labels=None):
        # Given a sequence of states (X) plot their mean and std on the same graph. For the Holton-Mass model, this means different zonal wind profiles.
        #key = "U"
        num_levels = len(qlevels)
        funlib = self.observable_function_library()
        Ua,Ub = funlib[key]["fun"](self.xst)
        units = funlib[key]["units"]
        q = self.q
        fig,ax = plt.subplots(figsize=(6,6),dpi=200) # Top panel for positive flux, bottom panel for negative flux
        # Plot a bunch of zonal wind profiles
        #N = len(cv_x)
        z = q['z_d'][1:-1]/1000
        handles = []
        ha, = ax.plot(units*Ua,z,color='blue',linestyle='dashed',linewidth=3,label=asymb)
        hb, = ax.plot(units*Ub,z,color='red',linestyle='dashed',linewidth=3,label=bsymb)
        handles += [ha,hb]
        if colors is None: colors = plt.cm.coolwarm(qlevels)
        if labels is None: labels = ["" for i in range(num_levels)]
        print("colors = {}, labels = {}".format(colors,labels))
        print("Before for loop: num_levels = {}, len(rflux) = {}, len(rflux_idx) = {}".format(num_levels,len(rflux),len(rflux_idx)))
        for i in range(num_levels):
            if len(rflux_idx[i]) > 0:
                U = funlib[key]["fun"](X[rflux_idx[i]])
                for zi in np.linspace(0,len(z)-1,10).astype(int): #range(len(z)):
                    order = np.argsort(U[:,zi])
                    rfzi = np.array(rflux[i])[order]
                    Uzi = U[order,zi]
                    # Restrict to places with flux > 10% of the max
                    sig_idx = np.where(np.abs(rfzi) > 0.2*np.max(np.abs(rfzi)))
                    rfzi = rfzi[sig_idx]
                    Uzi = Uzi[sig_idx]
                    # Make a histogram
                    hist,bin_edges = np.histogram(Uzi,weights=rfzi,density=False,bins=min(20,len(rfzi)))
                    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
                    # Normalize
                    hist *= 0.5*(z[1]-z[0])/np.max(np.abs(hist))
                    # Plot
                    ax.plot(units*bin_centers,z[zi]*np.ones(len(bin_centers)),color=colors[i],alpha=0.5)
                    ax.plot(units*bin_centers,z[zi]+hist,color=colors[i],linewidth=1)
        #ax.legend(handles=handles,prop={'size':13})
        ax.set_ylabel(r"$z$ [km]",fontdict=font)
        ax.set_xlabel("{} [{}]".format(funlib[key]['name'],funlib[key]['unit_symbol']),fontdict=font)
        xlim = ax.get_xlim()
        fmt_x = helper.generate_sci_fmt(xlim[0],xlim[1])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
        return fig,ax
    def plot_state_distribution(self,X,rflux,rflux_idx,qlevels,qsymbol,colors=None,key="U",labels=None,quantiles=[0.25,0.75],fig=None,ax=None):
        # Given a sequence of states (X) plot their mean and std on the same graph. For the Holton-Mass model, this means different zonal wind profiles.
        #key = "U"
        num_levels = len(qlevels)
        funlib = self.observable_function_library()
        Ua,Ub = funlib[key]["fun"](self.xst)
        units = funlib[key]["units"]
        q = self.q
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,6)) 
        # Plot a bunch of zonal wind profiles
        #N = len(cv_x)
        z = q['z_d'][1:-1]/1000
        handles = []
        ha, = ax.plot(units*Ua,z,color='blue',linestyle='dashed',linewidth=3,label=asymb)
        hb, = ax.plot(units*Ub,z,color='red',linestyle='dashed',linewidth=3,label=bsymb)
        handles += [ha,hb]
        if colors is None: colors = plt.cm.coolwarm(qlevels)
        if labels is None: labels = ["" for i in range(num_levels)]
        print("colors = {}, labels = {}".format(colors,labels))
        print("Before for loop: num_levels = {}, len(rflux) = {}, len(rflux_idx) = {}".format(num_levels,len(rflux),len(rflux_idx)))
        for i in range(num_levels):
            if len(rflux_idx[i]) > 0:
                U = funlib[key]["fun"](X[rflux_idx[i]])
                Umean = np.zeros(len(z))
                Umedian = np.zeros(len(z))
                Ulower = np.zeros(len(z))
                Uupper = np.zeros(len(z))
                for zi in range(len(z)):
                    order = np.argsort(U[:,zi])
                    rfzi = np.array(rflux[i])[order]
                    Uzi = U[order,zi]
                    cdf = np.cumsum(rfzi)
                    cdf *= 1.0/cdf[-1]
                    Ulower[zi] = Uzi[np.where(cdf > quantiles[0])[0][0]]
                    Uupper[zi] = Uzi[np.where(cdf > quantiles[1])[0][0]]
                    Umean[zi] = np.sum(Uzi*rfzi)/np.sum(rfzi)
                    Umedian[zi] = Uzi[np.where(cdf > 0.5)[0][0]]
                #Umean = np.mean(U,axis=0)
                #Ustd = np.std(U,axis=0)
                handle, = ax.plot(units*Umedian,z,color=colors[i],linewidth=3,label=labels[i])
                handles += [handle]
                ax.fill_betweenx(z,x1=units*Ulower,x2=units*Uupper,color=colors[i],alpha=0.5)
        ax.legend(handles=handles,prop={'size':13})
        ax.set_ylabel(r"$z$ [km]",fontdict=font)
        ax.set_xlabel("{} [{}]".format(funlib[key]['name'],funlib[key]['unit_symbol']),fontdict=font)
        xlim = ax.get_xlim()
        fmt_x = helper.generate_sci_fmt(xlim[0],xlim[1])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
        return fig,ax
    def plot_profile_evolution(self,prof,levels,func_key,fig=None,ax=None,clim=None,logscale=False):
        # Plot a sequence of profiles, smoothly, as in the least action path.
        print("prof.shape = {}".format(prof.shape))
        print("levels.shape = {}".format(levels.shape))
        funlib = self.observable_function_library()
        n = prof.shape[1]
        num_snaps = 100
        levels_interp = np.linspace(levels[0],levels[-1],num_snaps)
        prof_interp = np.zeros((num_snaps,n))
        for i in range(n):
            prof_interp[:,i] = interp1d(levels,prof[:,i],kind='cubic')(levels_interp)
        z = self.q['z_d'][1:-1]/1000
        lz,zl = np.meshgrid(levels_interp,z,indexing='ij')
        if fig is None or ax is None:
            fig,ax = plt.subplots()
        vmin,vmax = (None,None) if clim is None else clim
        eps = np.min(np.abs(prof))/2
        if logscale:
            im = ax.contourf(lz,zl,np.maximum(eps,prof_interp)*funlib[func_key]["units"],cmap=plt.cm.coolwarm,norm=mpl_colors.LogNorm(vmin=vmin,vmax=vmax))
        else:
            im = ax.contourf(lz,zl,prof_interp*funlib[func_key]["units"],cmap=plt.cm.coolwarm,vmin=vmin,vmax=vmax)
        ax.set_ylabel(r"$z$ [km]",fontdict=font)
        ax.set_xlabel(r"$-\eta_B^+$ [days]",fontdict=font)
        return fig,ax,im
    def plot_multiple_states(self,X,qlevels,qsymbol,colorlist=None,zorderlist=None,key="U",labellist=None):
    #def plot_zdep_family_weighted(self,cv_x,cv_a,cv_b,labels,weights=None,cv_name=None,colorlist=None,units=1.0,unit_symbol=""):
        # Given a sequence of states (X) plot them all on the same graph. For the Holton-Mass model, this means different zonal wind profiles.
        #key = "U"
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
        ax.plot(units*Ua,z,color='blue',linestyle='--',linewidth=4,zorder=np.max(zorderlist)+1)
        ax.plot(units*Ub,z,color='red',linestyle='--',linewidth=4,zorder=np.max(zorderlist)+1)
        if colorlist is None: colorlist = plt.cm.coolwarm(qlevels)
        if labellist is None: labellist = ["" for i in range(num_states)]
        handles = []
        for i in range(num_states):
            handle, = ax.plot(units*U[i],z,alpha=1.0,color=colorlist[i],zorder=zorderlist[i],linewidth=3,label=labellist[i])
            if len(labellist[i]) > 0: handles += [handle]
            #if i % len(np.unique(qlevels)) == 0: handles += [handle]
        ax.legend(handles=handles,prop={'size':13})
        ax.set_ylabel(r"$z$ [km]",fontdict=font)
        ax.set_xlabel("{} [{}]".format(funlib[key]['name'],funlib[key]['unit_symbol']),fontdict=font)
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
    ax[0].set_ylabel("Alitude [km]")
    ax[1].plot(model.xst[0,n:2*n],qnew['z_d'][1:-1]/1000,color='blue')
    ax[1].plot(model.xst[1,n:2*n],qnew['z_d'][1:-1]/1000,color='red')
    ax[1].set_xlabel("Im(Psi)")
    ax[1].set_ylabel("Alitude [km]")
    ax[2].plot(model.xst[0,2*n:3*n],qnew['z_d'][1:-1]/1000,color='blue')
    ax[2].plot(model.xst[1,2*n:3*n],qnew['z_d'][1:-1]/1000,color='red')
    ax[2].set_xlabel("U")
    ax[2].set_ylabel("Alitude [km]")
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
