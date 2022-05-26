import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 25
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
from hm_model import HoltonMassModel
import hm_params 
import os
from os.path import join,exists
import sys
savedir = "/scratch/jf4241/SHORT_holtonmass/results/2022-05-23/0"

physical_params,_ = hm_params.get_physical_params()
physical_params['hB_d'] = 55
xst = np.load("xst.npy")
model = HoltonMassModel(physical_params,xst)
q = model.q
z = q['z_d'][1:-1]/1000
np.save("xst",model.xst)

# Does az = a_z?
a = q['alpha'][1:-1]
az_findiff = (q['alpha'][2:] - q['alpha'][:-2])/(2*q['dz'])
az = q['alpha_z'][1:-1]
fig,ax = plt.subplots()
ha, = ax.plot(z,a,color='black',label=r"$\alpha(z)$")
haz, = ax.plot(z,az,color='cyan',label=r"$\partial_z\alpha(z)$ analytical")
haz_findiff, = ax.plot(z,az_findiff,color='red',label=r"$\partial_z\alpha(z)$ finite difference")
ax.legend(handles=[ha,haz,haz_findiff])
fig.savefig(join(savedir,"alphader"))
plt.close(fig)
sys.exit()


afrac = np.linspace(0,1,4)
bfrac = 1 - afrac
x = np.outer(afrac, model.xst[0]) + np.outer(bfrac, model.xst[1])
enstrophy_tendency,pvflux,pvgrad,diss,lhs = model.test_enstrophy_equation(x,lat=60,dt=0.001)
for i in range(len(x)):
    fig,ax = plt.subplots(ncols=2,nrows=5,figsize=(12,30))
    # Left column: d(enstrophy)/dt, vq, dq/dy, -dissipation plus everything else
    # Enstrophy tendency
    ax[0,0].plot(z,enstrophy_tendency[i],color='red')
    ax[0,0].set_title(r"$\frac{1}{2}\partial_t\overline{q'^2}$")
    # PV flux
    ax[1,0].plot(z,pvflux[i],color='black')
    ax[1,0].set_title(r"$\overline{v'q'}$")
    # PV gradient
    ax[2,0].plot(z,pvgrad[i],color='black')
    ax[2,0].set_title(r"$\partial_y\overline{q}$")
    # (PV flux) * (PV gradient)
    ax[3,0].plot(z,pvgrad[i]*pvflux[i],color='black')
    ax[3,0].set_title(r"$\overline{v'q'}\partial_y\overline{q}$")
    # Dissipation
    ax[4,0].plot(z,-diss[i],color='cyan')
    ax[4,0].set_title(r"$-$Dissipation")
    # Everything together
    ax[4,0].plot(z,enstrophy_tendency[i],color='red')
    ax[4,0].plot(z,pvflux[i]*pvgrad[i],color='black')
    ax[4,0].plot(z,lhs[i],color='gray',linestyle='--')
    # Right column: everything divided by dq/dy 
    # Enstrophy tendency / (PV gradient)
    ax[0,1].plot(z,enstrophy_tendency[i]/pvgrad[i],color='red')
    ax[0,1].set_title(r"$\frac{1}{2}\partial_t\overline{q'^2}/\partial_y\overline{q}$")
    # (PV flux) 
    ax[3,1].plot(z,pvflux[i],color='black')
    ax[3,1].set_title(r"$\overline{v'q'}$")
    # Dissipation
    ax[4,1].plot(z,-diss[i]/pvgrad[i],color='cyan')
    ax[4,1].set_title(r"$-$Dissipation")
    # Everything together
    ax[4,1].plot(z,enstrophy_tendency[i]/pvgrad[i],color='red')
    ax[4,1].set_title(r"$\overline{v'q'}$")
    ax[4,1].plot(z,pvflux[i],color='black')
    ax[4,1].plot(z,lhs[i]/pvgrad[i],color='gray',linestyle='--')
    # Invert the horizontal axis 
    for r in range(5):
        for c in range(2):
            ax[r,c].invert_xaxis()
    fig.savefig(join(savedir,f"balance_{i}"))
    plt.close(fig)
    
