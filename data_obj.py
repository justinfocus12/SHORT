# This is where the Data object lives

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')
matplotlib.rcParams['font.size'] = 17
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
        
def fmt(num,pos):
    return '{:.1e}'.format(num)

def both_grids(bounds,shp):
    # This time shp is the number of cells
    Nc = np.prod(shp-1)   # Number of centers Ne = np.prod(shp) # Number of edges
    center_grid = np.array(np.unravel_index(np.arange(Nc),shp-1)).T
    edge_grid = np.array(np.unravel_index(np.arange(Ne),shp)).T
    dx = (bounds[:,1] - bounds[:,0])/(shp - 1)
    center_grid = bounds[:,0] + dx * (center_grid + 0.5)
    edge_grid = bounds[:,0] + dx * edge_grid
    return center_grid,edge_grid,dx

class Data:
    def __init__(self,x_short,t_short,lag_time_seq):
        # bdy_dist just needs to be zero on the boundaries
        Nt,self.nshort,self.xdim = x_short.shape
        self.traj_length = len(lag_time_seq)
        self.X = np.zeros((self.nshort,self.traj_length,self.xdim))
        self.t_x = np.zeros(self.traj_length) 
        time_indices = np.zeros(self.traj_length,dtype=int)
        for i in range(self.traj_length):
            time_indices[i] = np.argmin(np.abs(t_short-lag_time_seq[i]))
            self.X[:,i,:] = x_short[time_indices[i]]
            self.t_x[i] = t_short[time_indices[i]]
        #self.X = x_short[0]
        #self.t_x = t_short[0]*np.ones(self.nshort)
        ##print("self.X.shape = {}".format(self.X.shape))
        #ti_y = np.argmin(np.abs(t_short-lag_time))
        #self.Y = x_short[ti_y]
        #self.t_y = t_short[ti_y]*np.ones(self.nshort)
        ##print("self.Y.shape = {}".format(self.Y.shape))
        #self.lag_time = lag_time
        #ti_xp = np.zeros(self.nshort, dtype=int)
        #ti_yp = ti_y*np.ones(self.nshort, dtype=int)
        #for j in range(min(ti_y,Nt)):
        #    db = bdy_dist(x_short[j])
        #    bdy_idx = np.where(db==0)[0]
        #    if len(bdy_idx) > 0:
        #        if j > 0:
        #            ti_yp[bdy_idx] = np.minimum(j,ti_yp[bdy_idx])
        #        if j < min(ti_y,Nt)-1:
        #            ti_xp[bdy_idx] = np.maximum(j, ti_xp[bdy_idx])
        #self.Yp = x_short[ti_yp,np.arange(self.nshort)] 
        #self.t_yp = t_short[ti_yp]
        ##print("std of t_yp = {}".format(np.std(self.t_yp)))
        ##print("Fraction of hits = {}".format(np.mean(self.t_yp < self.lag_time)))
        #self.Xp = x_short[ti_xp,np.arange(self.nshort)]
        #self.t_xp = t_short[ti_xp]
        ##print("self.Yp.shape = {}".format(self.Yp.shape))
        ##print("self.Xp.shape = {}".format(self.Xp.shape))
        ## Get all the distance arrays
        #self.bdy_dist_x = bdy_dist(self.X)
        #self.bdy_dist_y = bdy_dist(self.Y)
        #self.bdy_dist_xp = bdy_dist(self.Xp)
        #self.bdy_dist_yp = bdy_dist(self.Yp)
        #self.bdy_idx_x = np.where(self.bdy_dist_x==0)[0]
        #self.iidx_x = np.where(self.bdy_dist_x!=0)[0]
        #self.bdy_idx_y = np.where(self.bdy_dist_y==0)[0]
        #self.iidx_y = np.where(self.bdy_dist_y!=0)[0]
        #self.bdy_idx_xp = np.where(self.bdy_dist_xp==0)[0]
        #self.iidx_xp = np.where(self.bdy_dist_xp!=0)[0]
        #self.bdy_idx_yp = np.where(self.bdy_dist_yp==0)[0]
        #self.iidx_yp = np.where(self.bdy_dist_yp!=0)[0]
        return
    def concatenate_data(self,other):
        # fold in a whole nother dataset
        self.X = np.concatenate((self.X,other.X),axis=0)
        #self.Y = np.concatenate((self.Y,other.Y),axis=0)
        #self.Xp = np.concatenate((self.Xp,other.Xp),axis=0)
        #self.Yp = np.concatenate((self.Yp,other.Yp),axis=0)
        #self.bdy_idx_x = np.where(bdy_dist(self.X)==0)[0]
        #self.iidx_x = np.where(bdy_dist(self.X)!=0)[0]
        #self.bdy_idx_y = np.where(bdy_dist(self.Y)==0)[0]
        #self.iidx_y = np.where(bdy_dist(self.Y)!=0)[0]
        #self.t_x = np.concatenate((self.t_x,other.t_x))
        #self.t_y = np.concatenate((self.t_y,other.t_y))
        #self.t_xp = np.concatenate((self.t_xp,other.t_xp))
        #self.t_yp = np.concatenate((self.t_yp,other.t_yp))
        self.nshort += other.nshort
        #self.bdy_dist_x = bdy_dist(self.X)
        #self.bdy_dist_y = bdy_dist(self.Y)
        #self.bdy_dist_xp = bdy_dist(self.Xp)
        #self.bdy_dist_yp = bdy_dist(self.Yp)
        #self.bdy_idx_x = np.where(self.bdy_dist_x==0)[0]
        #self.iidx_x = np.where(self.bdy_dist_x!=0)[0]
        #self.bdy_idx_y = np.where(self.bdy_dist_y==0)[0]
        #self.iidx_y = np.where(self.bdy_dist_y!=0)[0]
        #self.bdy_idx_xp = np.where(self.bdy_dist_xp==0)[0]
        #self.iidx_xp = np.where(self.bdy_dist_xp!=0)[0]
        #self.bdy_idx_yp = np.where(self.bdy_dist_yp==0)[0]
        #self.iidx_yp = np.where(self.bdy_dist_yp!=0)[0]
        return
    def insert_boundaries(self,bdy_dist,lag_time_max=None):
        if lag_time_max is None: lag_time_max=self.t_x[-1]
        # Find the last-exit and first-entry points
        bdy_dist_x = bdy_dist(self.X.reshape((self.nshort*self.traj_length,self.xdim))).reshape((self.nshort,self.traj_length))
        self.last_entry_idx = np.zeros(self.nshort,dtype=int)
        ti_max = np.argmin(np.abs(lag_time_max - self.t_x))
        self.last_idx = ti_max*np.ones(self.nshort,dtype=int) # for yj
        self.first_exit_idx = ti_max*np.ones(self.nshort,dtype=int)
        for i in range(ti_max):
            db = bdy_dist(self.X[:,i,:])
            bidx = np.where(db==0)[0]
            if i < self.traj_length-1:
                self.last_entry_idx[bidx] = i
            if i > 0:
                self.first_exit_idx[bidx] = np.minimum(self.first_exit_idx[bidx],i)
        return
    def insert_boundaries_fwd(self,bdy_dist_x,tmin,tmax):
        if tmin > tmax: sys.exit("HEY! Make sure tmin < tmax in insert_boundaries_fwd")
        #bdy_dist_x = bdy_dist(self.X.reshape((self.nshort*self.traj_length,self.xdim))).reshape((self.nshort,self.traj_length))
        Nx,Nt = bdy_dist_x.shape
        ti_min = np.argmin(np.abs(tmin - self.t_x))
        ti_max = np.argmin(np.abs(tmax - self.t_x))
        self.base_idx_fwd = ti_min*np.ones(Nx,dtype=int)
        self.first_exit_idx_fwd = ti_max*np.ones(Nx,dtype=int)
        self.last_idx_fwd = ti_max*np.ones(Nx,dtype=int)
        for i in np.arange(ti_max-1,ti_min,-1):
            bidx = np.where(bdy_dist_x[:,i]==0)[0]
            self.first_exit_idx_fwd[bidx] = i
        return
    def insert_boundaries_bwd(self,bdy_dist_x,tmax,tmin):
        if tmin > tmax: 
            raise Exception("HEY! Make sure tmax > tmin in insert_boundaries_bwd")
        #bdy_dist_x = bdy_dist(self.X.reshape((self.nshort*self.traj_length,self.xdim))).reshape((self.nshort,self.traj_length))
        Nx,Nt = bdy_dist_x.shape
        ti_min = np.argmin(np.abs(tmin - self.t_x))
        ti_max = np.argmin(np.abs(tmax - self.t_x))
        self.base_idx_bwd = ti_max*np.ones(Nx,dtype=int)
        self.first_exit_idx_bwd = ti_min*np.ones(Nx,dtype=int)
        self.last_idx_bwd = ti_min*np.ones(Nx,dtype=int)
        for i in np.arange(ti_min+1,ti_max,1):
            bidx = np.where(bdy_dist_x[:,i]==0)[0]
            self.first_exit_idx_bwd[bidx] = i
        return
        
