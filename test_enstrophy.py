import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 25
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
from hm_model import HoltonMassModel,first_derivative,second_derivative
import hm_params 
import os
from os import mkdir
from os.path import join,exists
import sys
savedir = "/scratch/jf4241/SHORT_holtonmass/results/2022-05-23/1"
if not exists(savedir): mkdir(savedir)

physical_params,_ = hm_params.get_physical_params()
physical_params['hB_d'] = 55
xst = np.load("xst.npy")
model = HoltonMassModel(physical_params,xst)
q = model.q
z = q['z_d'][1:-1]/1000
n = q['Nz']-1
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


ez = np.exp(q['z'][1:-1])
ez2 = np.exp(q['z'][1:-1]/2)
afrac = np.linspace(0,1,4)
bfrac = 1 - afrac
x = np.outer(afrac, model.xst[0]) + np.outer(bfrac, model.xst[1])

# ------- Test the basic tendency ---------
X,Y,U = x[:,:n],x[:,n:2*n],x[:,2*n:]
U_lower = q['UR_0']
U_upper = 1.0/3*(4*U[:,-1] - U[:,-2] + 2*q['dz']*q['UR_z'][-1])
X_lower = q['Psi0']
X_upper = 0
Y_lower = 0
Y_upper = 0
Xz = first_derivative(X,X_lower,X_upper,q['dz'])
Yz = first_derivative(Y,Y_lower,Y_upper,q['dz'])
Uz = first_derivative(U,U_lower,U_upper,q['dz'])
Xzz = second_derivative(X,X_lower,X_upper,q['dz'])
Yzz = second_derivative(Y,Y_lower,Y_upper,q['dz'])
Uzz = second_derivative(U,U_lower,U_upper,q['dz'])
rhs = np.zeros((len(x),3*n))
# Re(Psi)
rhs[:,:n] = (a/4 - az/2)*X - az*Xz - a*Xzz + q['Gsq']*q['k']*q['beta']*Y
rhs[:,:n] -= q['k']*q['eps']*Y*(
    (q['k']**2*q['Gsq'] + 0.25)*U - Uz + Uzz
    )
rhs[:,:n] += q['k']*q['eps']*U*Yzz
# Im(Psi)
rhs[:,n:2*n] = (a/4 - az/2)*Y - az*Yz - a*Yzz - q['Gsq']*q['k']*q['beta']*X
rhs[:,n:2*n] += q['k']*q['eps']*X*(
        (q['k']**2*q['Gsq'] + 0.25)*U - Uz + Uzz
        )
rhs[:,n:2*n] -= q['k']*q['eps']*U*Xzz
# U 
rhs[:,2*n:] = (az - a)*q['UR_z'][1:-1] - a*q['UR_zz'][1:-1]
rhs[:,2*n:] -= (az-a)*Uz + a*Uzz
rhs[:,2*n:] += q['eps']*q['k']*q['l']**2/2*ez*(Y*Xzz - X*Yzz)
# Compute the tendencies 
dx_dt = model.drift_fun(x)
Xdot,Ydot,Udot = dx_dt[:,:n],dx_dt[:,n:2*n],dx_dt[:,2*n:]
Udot_upper = 1.0/3*(4*Udot[:,-1] - Udot[:,-2])
Xdot_zz = second_derivative(Xdot,0,0,q['dz'])
Ydot_zz = second_derivative(Ydot,0,0,q['dz'])
Udot_zz = second_derivative(Udot,0,Udot_upper,q['dz'])
print(f"Udot_upper.shape = {Udot_upper.shape}")
lhs = np.zeros((len(x),3*n))
lhs[:,:2*n] = -(q['Gsq']*(q['k']**2+q['l']**2) + 0.25)*dx_dt[:,:2*n]
lhs[:,:n] += second_derivative(Xdot,0,0,q['dz'])
lhs[:,n:2*n] += second_derivative(Ydot,0,0,q['dz'])
lhs[:,2*n:] = -q['Gsq']*q['l']**2*Udot 
lhs[:,2*n:] -= first_derivative(Udot,0,Udot_upper,q['dz']) 
lhs[:,2*n:] += second_derivative(Udot,0,Udot_upper,q['dz'])
# Plot 
fig,ax = plt.subplots(nrows=len(x),ncols=3,figsize=(18,len(x)*6))
for i in range(len(x)):
    ax[i,0].plot(z,lhs[i,:n],color='lightskyblue',linewidth=5)
    ax[i,0].plot(z,rhs[i,:n],color='red',linestyle='--')
    ax[i,1].plot(z,lhs[i,n:2*n],color='lightskyblue',linewidth=5)
    ax[i,1].plot(z,rhs[i,n:2*n],color='red',linestyle='--')
    ax[i,2].plot(z,lhs[i,2*n:],color='lightskyblue',linewidth=5)
    ax[i,2].plot(z,rhs[i,2*n:],color='red',linestyle='--')
fig.savefig(join(savedir,"tendency_test"))
plt.close(fig)
# -----------------------------------------

# ---------- Test the tendency of the rearranged equation for X -------------
delta = q['Gsq']*(q['k']**2 + q['l']**2) + 1.0/4
ens_term = np.zeros((len(x),2*n))
ens_term[:,:n] = -delta*Xdot + Xdot_zz - q['k']*q['eps']*U*(-delta*Y + Yzz)
ens_term[:,n:2*n] = -delta*Ydot + Ydot_zz + q['k']*q['eps']*U*(-delta*X + Xzz)
adv_term = np.zeros((len(x),2*n))
adv_term[:,:n] = -q['k']*Y*(q['Gsq']*q['beta'] + q['eps']*(q['Gsq']*q['l']**2*U + Uz - Uzz))
adv_term[:,n:2*n] = q['k']*X*(q['Gsq']*q['beta'] + q['eps']*(q['Gsq']*q['l']**2*U + Uz - Uzz))
dis_term = np.zeros((len(x),2*n))
dis_term[:,:n] = (a/4 - az/2)*X - az*Xz - a*Xzz
dis_term[:,n:2*n] = (a/4 - az/2)*Y - az*Yz - a*Yzz
# Plot 
fig,ax = plt.subplots(nrows=len(x),ncols=2,figsize=(12,len(x)*6))
for i in range(len(x)):
    for j in range(2):
        ax[i,j].plot(z,ens_term[i,j*n:(j+1)*n],color='lightskyblue')
        ax[i,j].plot(z,adv_term[i,j*n:(j+1)*n],color='springgreen')
        ax[i,j].plot(z,-dis_term[i,j*n:(j+1)*n],color='red')
        ax[i,j].plot(z,ens_term[i,j*n:(j+1)*n]+adv_term[i,j*n:(j+1)*n]-dis_term[i,j*n:(j+1)*n],color='gray',linestyle='--')
fig.savefig(join(savedir,"tendency_test_rearranged"))
plt.close(fig)
# ---------------------------------------------------------------------


enstrophy_tendency,pvflux,pvgrad,diss,lhs = model.test_enstrophy_equation(x,lat=60,dt=0.00001)
#enstrophy_tendency *= ez
#pvflux *= ez2
#pvgrad *= ez2
#diss *= ez
#lhs *= ez
for i in range(len(x)):
    fig,ax = plt.subplots(ncols=2,nrows=5,figsize=(12,30))
    # Left column: d(enstrophy)/dt, vq, dq/dy, -dissipation plus everything else
    # Enstrophy tendency
    ax[0,0].plot(z,enstrophy_tendency[i],color='lightskyblue')
    ax[0,0].set_title(r"$\frac{1}{2}\partial_t\overline{q'^2}$")
    # PV flux
    ax[1,0].plot(z,pvflux[i],color='green')
    ax[1,0].set_title(r"$\overline{v'q'}$")
    # PV gradient
    ax[2,0].plot(z,pvgrad[i],color='green')
    ax[2,0].set_title(r"$\partial_y\overline{q}$")
    # (PV flux) * (PV gradient)
    ax[3,0].plot(z,pvgrad[i]*pvflux[i],color='green')
    ax[3,0].set_title(r"$\overline{v'q'}\partial_y\overline{q}$")
    # Dissipation
    ax[4,0].plot(z,-diss[i],color='red')
    ax[4,0].set_title(r"$-$Dissipation")
    # Everything together
    ax[4,0].plot(z,enstrophy_tendency[i],color='lightskyblue')
    ax[4,0].plot(z,pvflux[i]*pvgrad[i],color='green')
    ax[4,0].plot(z,lhs[i],color='gray',linestyle='--')
    # Right column: everything divided by dq/dy 
    # Enstrophy tendency / (PV gradient)
    ax[0,1].plot(z,enstrophy_tendency[i]/pvgrad[i],color='lightskyblue')
    ax[0,1].set_title(r"$\frac{1}{2}\partial_t\overline{q'^2}/\partial_y\overline{q}$")
    # (PV flux) 
    ax[3,1].plot(z,pvflux[i],color='green')
    ax[3,1].set_title(r"$\overline{v'q'}$")
    # Dissipation
    ax[4,1].plot(z,-diss[i]/pvgrad[i],color='red')
    ax[4,1].set_title(r"$-$Dissipation/$\partial_y\overline{q}$")
    # Everything together
    ax[4,1].plot(z,enstrophy_tendency[i]/pvgrad[i],color='lightskyblue')
    ax[4,1].plot(z,pvflux[i],color='green')
    ax[4,1].plot(z,lhs[i]/pvgrad[i],color='gray',linestyle='--')
    # Invert the horizontal axis 
    for r in range(5):
        for c in range(2):
            ax[r,c].invert_xaxis()
    fig.savefig(join(savedir,f"balance_{i}"))
    plt.close(fig)
    
