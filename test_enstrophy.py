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
savedir = "/scratch/jf4241/SHORT_holtonmass/results/2022-05-23/0"

physical_params,_ = hm_params.get_physical_params()
physical_params['hB_d'] = 55
xst = np.load("xst.npy")
model = HoltonMassModel(physical_params,xst)
q = model.q
np.save("xst",model.xst)

afrac = np.linspace(0,1,4)
bfrac = 1 - afrac
x = np.outer(afrac, model.xst[0]) + np.outer(bfrac, model.xst[1])
enstrophy_tendency,pvflux,pvgrad,diss,lhs = model.test_enstrophy_equation(x,lat=60,dt=0.001)
for i in range(len(x)):
    fig,ax = plt.subplots(ncols=3,nrows=2,figsize=(18,12))
    # Enstrophy tendency
    for axi in [ax[0,0],ax[1,1]]:
        axi.plot(enstrophy_tendency[i],q['z'][1:-1],color='red')
    ax[0,0].set_title(r"$\partial_t\frac{\overline{q'^2}}{2}$")
    # PV flux
    ax[0,1].plot(pvflux[i],q['z'][1:-1],color='black')
    ax[0,1].set_title(r"$\overline{v'q'}$")
    # PV gradient
    ax[0,2].plot(pvgrad[i],q['z'][1:-1],color='black')
    ax[0,2].set_title(r"$\partial_y\overline{q}$")
    # Their product
    ax[1,1].plot(pvgrad[i]*pvflux[i],q['z'][1:-1],color='black')
    ax[1,1].set_title(r"$\partial_y\overline{q}\overline{v'q'}$")
    # Dissipation
    for axi in [ax[1,0],ax[1,1]]:
        axi.plot(-diss[i],q['z'][1:-1],color='cyan')
    ax[1,0].set_title(r"$-$Dissipation")
    # All together
    ax[1,1].plot(lhs[i],q['z'][1:-1],color='grey',linestyle='--')
    ax[1,1].set_title("LHS")
    fig.savefig(join(savedir,f"balance_{i}"))
    plt.close(fig)
    
# Plot the Yoden form of dissipation
diss_yoden = diss/pvgrad 
fig,ax = plt.subplots()
ha, = ax.plot(q['z'][1:-1],diss_yoden[-1],color='cyan',label='A')
hb, = ax.plot(q['z'][1:-1],diss_yoden[0],color='black',label='B')
h, = ax.plot(q['z'][1:-1],diss[0],color='black',linestyle='dotted',label='B without division')
ax.legend(handles=[ha,hb,h])
ax.set_title("Yoden dissipation")
fig.savefig(join(savedir,"diss_yoden"))
plt.close(fig)
