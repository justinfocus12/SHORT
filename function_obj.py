import numpy as np
from numpy import save,load
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['font.size'] = 17
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from scipy.interpolate import interp1d
import scipy.sparse as sps
from scipy.sparse import linalg as sps_linalg
import scipy.linalg as scipy_linalg
from importlib import reload
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import PCA
from abc import ABC,abstractmethod
import time
import queue
import sys
import os
from os.path import join,exists
codefolder = "/home/jf4241/dgaf2"
os.chdir(codefolder)
from hier_cluster_obj import nested_kmeans,nested_kmeans_predict_batch

# TODO: turn this all time-dependent later. For now, keep it time-homogeneous.

class Function(ABC):
    # General kind of function that could be a neural network or a Gaussian process or a linear basis...
    def __init__(self,function_type):
        self.function_type = function_type # linear_basis, gaussian_process, neural_network 
        super().__init__()
        return
    @abstractmethod
    def evaluate_function(self,data):
        # This should be done after the parameters are set, or at least initialized
        # X.shape = (Nx,xdim) (possibly from a flattened array)
        pass
    @abstractmethod
    def feynman_kac_rhs(self,data,src_fun,dirn=1):
        Nx,Nt,xdim = data.X.shape
        HX = (src_fun(data.X.reshape((Nx*Nt,xdim)))).reshape((Nx,Nt))
        R = np.zeros(Nx)
        for i in range(Nt-1):
            if dirn==1:
                dt = data.t_x[i+1] - data.t_x[i]
                if dt<0: sys.exit("Ummm dt<0 forward")
                R -= 0.5*(HX[:,i] + HX[:,i+1]) * dt * (data.first_exit_idx > i)
            else:
                dt = data.t_x[-i-1] - data.t_x[-i-2]
                if dt<0: sys.exit("Ummm dt<0 backward. i={}, data.t_x={}, data.t_x[-i]={}, data.t_x[-i-1]={}".format(i,data.t_x,data.t_x[-i],data.t_x[-i-1]))
                R -= 0.5*(HX[:,-i-1] + HX[:,-i-2]) * dt * (data.last_entry_idx < data.traj_length-i)
        return R
    def feynman_kac_lhs_FX(self,data,unk_fun,unk_fun_dim):
        # First part of the left-hand side is to compute just F itself. This is expensive! Afterward we get LF and VF easily
        Nx,Nt,xdim = data.X.shape
        #FX = unk_fun(data).reshape((Nx,Nt,unk_fun_dim))
        FX = (unk_fun(data.X.reshape((Nx*Nt,xdim)))).reshape((Nx,Nt,unk_fun_dim))
        return FX
    def feynman_kac_lhs_LF(self,data,FX,dirn=1):
        # Second part of the left-hand side is to compute LF. (The stopped version.)
        Nx,Nt,xdim = data.X.shape
        if dirn == 1:
            LF = FX[np.arange(Nx),data.first_exit_idx] - FX[:,0]
        else:
            LF = FX[np.arange(Nx),data.last_entry_idx] - FX[:,-1]
        return LF
    def feynman_kac_lhs_VX(self,data,pot_fun):
        Nx,Nt,xdim = data.X.shape
        VX = (pot_fun(data.X.reshape((Nx*Nt,xdim)))).reshape((Nx,Nt))
        return VX
    def feynman_kac_lhs_VF(self,data,VX,FX,Fdim,dirn=1):
        Nx,Nt,xdim = data.X.shape
        VF = np.zeros((Nx,Fdim))
        #print("About to compute VF, whose shape is {}".format(VF.shape))
        for i in range(Nt-1):
            if dirn==1:
                dt = data.t_x[i+1] - data.t_x[i]
                VF += 0.5*((VX[:,i]*FX[:,i].T + VX[:,i+1]*FX[:,i+1].T) * dt * (data.first_exit_idx > i)).T
            else:
                dt = data.t_x[-i-1] - data.t_x[-i-2]
                VF += 0.5*((VX[:,-i-1]*FX[:,-i-1].T + VX[:,-i-2]*FX[:,-i-2].T) * dt * (data.last_entry_idx < data.traj_length-i)).T
        return VF
    def feynman_kac_lhs(self,data,unk_fun,unk_fun_dim,pot_fun,dirn=1):
        FX = self.feynman_kac_lhs_FX(data,unk_fun,unk_fun_dim)
        LF = self.feynman_kac_lhs_LF(data,FX,dirn=dirn)
        VX = self.feynman_kac_lhs_VX(data,pot_fun)
        VF = self.feynman_kac_lhs_VF(data,VX,FX,unk_fun_dim,dirn=dirn)
        return LF,VF,FX
    @abstractmethod
    def fit_data(self,X,bdy_dist):
        # This is implemented differently depending on the type, and may or may not be data-dependent
        # 1. LinearBasis: define basis functions (e.g. form cluster centers for MSM, or axes for PCA)
        # 2. GaussianProcess: define mean and covariance functions
        # 3. NeuralNetwork: define architecture and activation functions
        # In all cases, get the capability to evaluate new points
        # bndy_dist is a function of x and t
        #@Nx,Nt,xdim = data.X.shape
        #@N = Nx*Nt
        #@X = data.X.reshape((N,xdim))
        N,xdim = X.shape
        # Now cluster
        bdy_dist_x = bdy_dist(X)
        bdy_idx = np.where(bdy_dist_x==0)[0]
        iidx = np.setdiff1d(np.arange(N),bdy_idx)
        print("len(bdy_idx) = {}".format(len(bdy_idx)))
        print("len(iidx) = {}".format(len(iidx)))
        self.fit_data_flat(X,iidx,bdy_idx,bdy_dist_x)
        return
    @abstractmethod
    def fit_data_flat(self,X,iidx,bdy_idx,bdy_dist_x):
        pass
    @abstractmethod
    def solve_boundary_value_problem(self,data,bdy_dist,bdy_fun):
        # Solve the boundary value problem given a dataset of short trajectories (perhaps with different lengths and resolutions) by optimizing parameters. 
        # 1. LinearBasis: invert the matrix of basis function evaluations
        # 2. GaussianProcess: invert the correlation matrix to derive the posterior mean and covariance
        # 3. NeuralNetwork: optimize weights with SGD of some kind
        pass

# Now implement the various instances
class LinearBasis(Function):
    def __init__(self,basis_size,basis_type):
        self.basis_size = basis_size
        self.basis_type = basis_type
        super().__init__('LinearBasis')
    def feynman_kac_rhs(self,*args,**kwargs):
        return super().feynman_kac_rhs(*args,**kwargs)
    def feynman_kac_lhs(self,*args,**kwargs):
        return super().feynman_kac_lhs(*args,**kwargs)
    def evaluate_function(self,data):
        # First evaluate the basis functions, then sum them together according to the coefficients (which have already been determined when this function is called)
        Nx,Nt,xdim = data.X.shape
        F = (self.bdy_fun(data.X.reshape((Nx*Nt,xdim)))).reshape((Nx,Nt))
        F += self.evaluate_basis_functions(data).dot(self.coeffs)
        return
    @abstractmethod
    def fit_data(self,data,bdy_dist):
        return super().fit_data(data,bdy_dist)
    def fit_data_flat(self,X,iidx,bdy_idx,bdy_dist_x):
        pass
    def compute_stationary_density(self,data):
        # Keep it simple, stupid
        bdy_dist = lambda x: np.ones(len(x)) # no boundaries
        data.insert_boundaries(bdy_dist)
        pot_fun = lambda x: np.zeros(len(x))
        def unk_fun(X):
            return self.evaluate_basis_functions(X,bdy_dist,const_fun_flag=True)
        print("About to compute the Feynman-Kac LHS")
        Lphi,Vphi,phi = self.feynman_kac_lhs(data,unk_fun,self.basis_size,pot_fun)
        phi_Lphi = phi[:,0].T.dot(Lphi)
        #sys.exit("phi_Lphi.dot(1) = {}".format(phi_Lphi.dot(np.ones(phi_Lphi.shape[1]))))
        Q,R = np.linalg.qr(phi_Lphi,mode='complete')
        print("Last two diags of R = {}".format(np.diag(R)[-2:]))
        v_phi = Q[:,-1]
        v_x = phi[:,0].dot(v_phi)
        return v_x
    def solve_damage_function_moments_multiple(self,data,bdy_dist,bdy_fun_list,dam_fun_list,dirn=1,weights=None,num_moments=1):
        Nx,Nt,xdim = data.X.shape
        num_bvp = len(bdy_fun_list)
        # No source function
        src_fun = lambda x: np.zeros(len(x))
        # All linear basis functions homogenize etc. in the same way, so this one can be implemented at this level
        # bdy_fun is also known as the guess fun
        # Insert boundaries
        data.insert_boundaries(bdy_dist)
        # Fit the data
        lag_idx = 0 if dirn==1 else data.traj_length-1
        print("Inside solve_damage_function_moments_multiple: about to fit data forward")
        print("data.X[:,lag_idx]: min={}, max={}, mean={}, std={}".format(np.min(data.X[:,lag_idx],0),np.max(data.X[:,lag_idx],0),np.mean(data.X[:,lag_idx],0),np.std(data.X[:,lag_idx],0)))
        self.fit_data(data.X[:,lag_idx],bdy_dist)
        print("Finished fitting data forward")
        if weights is None: weights = np.ones(data.nshort)/data.nshort
        def unk_fun(X):
            return self.evaluate_basis_functions(X,bdy_dist,const_fun_flag=False)
        # MOST EXPENSIVE STEP HERE--REPEAT AS LITTLE AS POSSIBLE
        phi = self.feynman_kac_lhs_FX(data,unk_fun,self.basis_size)
        # Maybe a little expensive hereafter, but not much
        Lphi = self.feynman_kac_lhs_LF(data,phi,dirn=dirn)
        A = (phi[:,lag_idx].T*weights).dot(Lphi)
        # TODO: compute a loss function that applies especially to Lphi
        print("In MOM: \n\tdetA = {}".format(np.linalg.det(A)))
        Ai = np.linalg.inv(A) # Sorrynotsorry
        # That matrix will be used for all the given boundary functions and damage functions.
        F = np.zeros((num_bvp,num_moments+1,Nx,Nt))
        smoothed_residual = np.zeros((num_bvp,Nx))
        Pay = np.zeros((num_bvp,Nx,Nt))
        for i in range(num_bvp):
            print("Starting bvp {} out of {}".format(i,num_bvp))
            pot_fun = lambda x: -dam_fun_list[i](x)
            tt = time.time()
            VX = self.feynman_kac_lhs_VX(data,pot_fun)
            print("feynman_kac_lhs_VX time = {:3.3e}".format(time.time()-tt))
            Pay[i] = -VX
            tt = time.time()
            Vphi = self.feynman_kac_lhs_VF(data,VX,phi,self.basis_size,dirn=dirn)
            print("feynman_kac_lhs_VF time = {:3.3e}".format(time.time()-tt))
            tt = time.time()
            Lbdy,Vbdy,bdy = self.feynman_kac_lhs(data,bdy_fun_list[i],1,pot_fun,dirn=dirn)
            print("feynman_kac_lhs time = {:3.3e}".format(time.time()-tt))
            B = -(phi[:,lag_idx].T*weights).dot(Vphi)
            c = ((phi[:,lag_idx].T*weights).dot(-Lbdy.flatten()))
            d = -((phi[:,lag_idx].T*weights).dot(-Vbdy)).flatten()
            AiB = Ai.dot(B)
            Aid = Ai.dot(d)
            Aic = Ai.dot(c)
            moments = np.zeros((num_moments+1,self.basis_size))
            moments[0] = Aic
            # Calculate the residual
            residual = (Lphi.dot(moments[0]) + Lbdy.flatten())**2
            smoothed_residual[i] = phi[:,lag_idx].dot(
                    (phi[:,lag_idx].T.dot(residual**2))/
                    (phi[:,lag_idx].T.dot(np.ones(data.nshort))))
            # This should give heavier weight to regions with a large mean-square residual
            moments[1] = Ai.dot(d - B.dot(moments[0]))
            for j in range(2,num_moments+1):
                moments[j] = -j*AiB.dot(moments[j-1])
                print("moments[{}]: min={}, max={}, mean={}, std={}".format(j,np.min(moments[j]),np.max(moments[j]),np.mean(moments[j]),np.std(moments[j])))
            # Now convert 
            tt = time.time()
            F[i,0] = bdy.reshape((Nx,Nt)) # Only the zeroth has a nontrivial boundary condition
            #for k in range(Nt):
            #   F[i,:,:,k] += phi[:,k,:].dot(moments)
            #  (nmom+1)x(Nx) (Nx)x(bs)   (nmom+1)x(bs)
            #    F[i,:,:,k] += moments.dot(phi[:,k,:].T)
            #  (nmom+1)x(Nx) (nmom+1)x(bs)  (bs)x(Nx)
            # ---------------- old but works--------------------
            for j in range(num_moments+1):
                for k in range(Nt):
                    F[i,j,:,k] += phi[:,k,:].dot(moments[j])
            #        (Nx)         (Nx)x(bs)    (bs)
            # --------------------------------------------------
            print("basis->data time: {:3.3e}".format(time.time()-tt))
        return F,Pay,smoothed_residual
    def solve_damage_function_moments(self,data,bdy_dist,bdy_fun,src_fun,pot_fun,dirn=1,weights=None,num_moments=1):
        # All linear basis functions homogenize etc. in the same way, so this one can be implemented at this level
        # bdy_fun is also known as the guess fun
        # Insert boundaries
        data.insert_boundaries(bdy_dist)
        if weights is None: weights = np.ones(data.nshort)/data.nshort
        def unk_fun(X):
            return self.evaluate_basis_functions(X,bdy_dist,const_fun_flag=False)
        Lphi,Vphi,phi = self.feynman_kac_lhs(data,unk_fun,self.basis_size,pot_fun,dirn=dirn)
        Lbdy,Vbdy,bdy = self.feynman_kac_lhs(data,bdy_fun,1,pot_fun,dirn=dirn)
        R = self.feynman_kac_rhs(data,src_fun,dirn=dirn)
        lag_idx = 0 if dirn==1 else data.traj_length-1
        # Split apart the matrices
        A = (phi[:,lag_idx].T*weights).dot(Lphi)
        #print("In MOM: A=\n{}".format(A))
        #print("In MOM. Are ingredients of A bad? \n\tphi rowsums in ({},{})\n\tLphi rowsums in ({},{})".format(np.min(phi.sum(1)),np.max(phi.sum(1)),np.min(np.abs(Lphi.sum(1))),np.max(np.abs(Lphi.sum(1)))))
        #if np.abs(np.linalg.det(A)) < 1e-6: sys.exit("DOH! Singular A matrix for moment")
        print("In MOM: \n\tdetA = {}".format(np.linalg.det(A)))
        B = -(phi[:,lag_idx].T*weights).dot(Vphi)
        C = ((phi[:,lag_idx].T*weights).dot(R - Lbdy.flatten()))
        print("R.shape = {}".format(R.shape))
        print("Lbdy.shape = {}".format(Lbdy.shape))
        print("C.shape = {}".format(C.shape))
        D = -((phi[:,lag_idx].T*weights).dot(-Vbdy)).flatten()
        Ai = np.linalg.inv(A)
        # Solve for coefficients at zero (which are the coefficients for the committor) 
        AiD = Ai.dot(D)
        AiB = Ai.dot(B)
        AiC = Ai.dot(C)
        moments = np.zeros((num_moments+1,self.basis_size))
        moments[0] = AiC
        moments[1] = Ai.dot(D - B.dot(moments[0]))
        for i in range(2,num_moments+1):
            moments[i] = -i*AiB.dot(moments[i-1])
            print("moments[{}]: min={}, max={}, mean={}, std={}".format(i,np.min(moments[i]),np.max(moments[i]),np.mean(moments[i]),np.std(moments[i])))
            #sys.exit()
        # Now convert 
        Nx,Nt,_ = data.X.shape
        moments_x = np.zeros((num_moments+1,Nx,Nt))
        moments_x[0] = bdy.reshape((Nx,Nt))
        for i in range(num_moments+1):
            #moments_x[i] = bdy.reshape((Nx,Nt))
            for j in range(Nt):
                moments_x[i,:,j] += phi[:,j].dot(moments[i])
            #moments_x[i] *= bdy.reshape((Nx,Nt))
        return moments_x # This is just the mean. We can easily go farther
    def solve_boundary_value_problem(self,data,bdy_dist,bdy_fun,src_fun,pot_fun,dirn=1,weights=None):
        # All linear basis functions homogenize etc. in the same way, so this one can be implemented at this level
        # bdy_fun is also known as the guess fun
        # Insert boundaries
        data.insert_boundaries(bdy_dist)
        if weights is None: weights = np.ones(data.nshort)/data.nshort
        def unk_fun(X):
            return self.evaluate_basis_functions(X,bdy_dist,const_fun_flag=False)
        Lphi,Vphi,phi = self.feynman_kac_lhs(data,unk_fun,self.basis_size,pot_fun,dirn=dirn)
        Lbdy,Vbdy,bdy = self.feynman_kac_lhs(data,bdy_fun,1,pot_fun,dirn=dirn)
        R = self.feynman_kac_rhs(data,src_fun,dirn=dirn)
        lag_idx = 0 if dirn==1 else data.traj_length-1
        phi_lhs = (phi[:,lag_idx].T*weights).dot(Lphi - Vphi)
        A = (phi[:,lag_idx].T*weights).dot(Lphi)
        #print("In BVP: A=\n{}".format(A))
        #print("In BVP. Are ingredients of A bad? \n\tphi rowsums in ({},{})\n\tLphi rowsums in ({},{})".format(np.min(phi.sum(1)),np.max(phi.sum(1)),np.min(np.abs(Lphi.sum(1))),np.max(np.abs(Lphi.sum(1)))))
        detA = np.linalg.det(A)
        detLHS = np.linalg.det(phi_lhs)
        print("In BVP: \n\tdetA = {}, detLHS = {}".format(detA,detLHS))
        phi_rhs = (phi[:,lag_idx].T*weights).dot(R - (Lbdy-Vbdy).flatten())
        self.coeffs = np.linalg.solve(phi_lhs,phi_rhs)
        Nx,Nt,_ = data.X.shape
        u = bdy.reshape((Nx,Nt))
        for i in range(data.traj_length):
            u[:,i] += phi[:,i].dot(self.coeffs) 
        self.bdy_fun = bdy_fun
        return u
    @abstractmethod
    def evaluate_basis_functions(self,X,bdy_dist,const_fun_flag=False):
        Nx,xdim = X.shape
        N = Nx
        #X = data.X.reshape((N,xdim))
        # Now cluster
        bdy_dist_x = bdy_dist(X)
        bdy_idx = np.where(bdy_dist_x==0)[0]
        iidx = np.setdiff1d(np.arange(N),bdy_idx)
        phi = self.evaluate_basis_functions_flat(X,iidx,bdy_idx,bdy_dist_x)
        #phi = phi.reshape((Nx,Nt,self.basis_size))
        return phi
    @abstractmethod
    def evaluate_basis_functions_flat(self,X,iidx,bdy_idx,bdy_dist_x):
        pass

# Now subclasses of LinearBasis
class MSMBasis(LinearBasis):
    def __init__(self,algo_params): #,basis_size,max_clust_per_level=200,min_clust_size=10):
        self.max_clust_per_level = algo_params['max_clust_per_level']
        self.min_clust_size = algo_params['min_clust_size']
        return super().__init__(algo_params['basis_size'],'MSM')
    def feynman_kac_rhs(self,*args,**kwargs):
        return super().feynman_kac_rhs(*args,**kwargs)
    def feynman_kac_lhs(self,*args,**kwargs):
        return super().feynman_kac_lhs(*args,**kwargs)
    def evaluate_function(self,*args,**kwargs):
        return super().evaluate_function(*args,**kwargs)
    def solve_boundary_value_problem(self,*args,**kwargs):
        return super().solve_boundary_value_problem(*args,**kwargs)
    def fit_data(self,*args,**kwargs):
        return super().fit_data(*args,**kwargs)
    def fit_data_flat(self,X,iidx,bdy_idx,bdy_dist_x):
        N,xdim = X.shape
        Ni = len(iidx)
        _,_,self.kmeans,self.centers = nested_kmeans(X[iidx],self.basis_size,mcpl=self.max_clust_per_level,min_clust_size=self.min_clust_size)
        return
    def evaluate_basis_functions(self,*args,**kwargs):
        return super().evaluate_basis_functions(*args,**kwargs)
    def evaluate_basis_functions_flat(self,X,iidx,bdy_idx,bdy_dist_x):
        # The constant function is ALWAYS in the span of MSM
        N,xdim = X.shape
        phi = np.zeros((N,self.basis_size))
        if len(iidx) > 0:
            _,global_addresses = nested_kmeans_predict_batch(X[iidx],self.kmeans)
            phi[iidx,global_addresses] = 1.0
        phi_colsums = np.sum(phi,0)
        phi_rowsums = np.sum(phi,1)
        print("phi colsums range = ({},{})".format(np.min(phi_colsums),np.max(phi_colsums)))
        print("phi rowsums range = ({},{})".format(np.min(phi_rowsums),np.max(phi_rowsums)))
        return phi

class PCABasis(LinearBasis):
    def __init__(self,basis_size,xdim):
        self.basis_size = min(basis_size,xdim)
        super().__init__(self.basis_size,'PCA')
        return
    def feynman_kac_rhs(self,*args,**kwargs):
        return super().feynman_kac_rhs(*args,**kwargs)
    def feynman_kac_lhs(self,*args,**kwargs):
        return super().feynman_kac_lhs(*args,**kwargs)
    def evaluate_function(self,*args,**kwargs):
        return super().evaluate_function(*args,**kwargs)
    def solve_boundary_value_problem(self,*args,**kwargs):
        return super().solve_boundary_value_problem(*args,**kwargs)
    def fit_data(self,*args,**kwargs):
        return super().fit_data(*args,**kwargs)
    def fit_data_flat(self,X,iidx,bdy_idx,bdy_dist_x):
        N,xdim = X.shape
        # do PCA on coordinates, then homogenize
        # Use coordinate functions a la John
        homogenizer = bdy_dist_x**2
        _,S,Vh = scipy_linalg.svd((X.T*homogenizer).T,full_matrices=False)
        self.V = Vh[:self.basis_size].T # basis_size x dimension of CV space
        self.S = S[:self.basis_size]
        return
    def evaluate_basis_functions(self,*args,**kwargs):
        return super().evaluate_basis_functions(*args,**kwargs)
    def evaluate_basis_functions_flat(self,X,iidx,bdy_idx,bdy_dist_x,const_fun_flag=False):
        # Now cluster
        homogenizer = bdy_dist_x**2
        coords = (Xt.T*homogenizer).T
        phi = coords.dot(self.V) / self.S # Nx
        if const_fun_flag:
            phi[:,-1] = 1.0/np.sqrt(len(x))
        return phi
