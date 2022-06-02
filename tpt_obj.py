# This file collects stuff specific to TPT and the Holton-Mass model
import numpy as np
from numpy import save,load
import scipy
from scipy.stats import describe
from scipy import special
from scipy.interpolate import splrep,splev
import pandas as pd
from sklearn import linear_model
import matplotlib
matplotlib.use('AGG')
#matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'monospace', 'size': 8}
medfont = {'family': 'monospace', 'size': 13}
font = {'family': 'monospace', 'size': 18}
ffont = {'family': 'monospace', 'size': 22}
bigfont = {'family': 'monospace', 'size': 30}
bbigfont = {'family': 'monospace', 'size': 40}
giantfont = {'family': 'monospace', 'size': 80}
ggiantfont = {'family': 'monospace', 'size': 120}
import pickle
import sys
import subprocess
import os
from os import mkdir
from os.path import join,exists
codefolder = "/home/jf4241/dgaf2"
os.chdir(codefolder)
from data_obj import Data
import helper

def fmt(num,pos):
    return '{:.1f}'.format(num)
def fmt2(num,pos):
    return '{:.2f}'.format(num)
def sci_fmt(num,lim):
    return '{:.1e}'.format(num)
def sci_fmt_short(num,lim):
    return '{:.0e}'.format(num)

# Symbols for A and B
asymb = r"$\mathbf{a}$"
bsymb = r"$\mathbf{b}$"

class TPT:
    def __init__(self,algo_params,physical_param_folder,long_simfolder,short_simfolder,savefolder):
        # adist, bdist, etc. will come from the model object only
        # type of function: should only come from the function object
        self.physical_param_folder = physical_param_folder
        self.nshort = algo_params['nshort']
        self.lag_time_current = algo_params['lag_time_current']
        self.lag_time_current_display = algo_params['lag_time_current_display']
        self.lag_time_seq = algo_params['lag_time_seq']
        self.num_moments = algo_params['num_moments']
        self.long_simfolder = long_simfolder
        self.short_simfolder = short_simfolder
        self.savefolder = savefolder
        return
    def label_x_long(self,model):
        # Big change: instead of just measuring time, we're also going to measure every integral quantity of interest given by the model
        # Keep track of integrals from and to AUB. When empirically measuring the moments, multiply by from_label and to_label
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        Nt = len(t_long)
        dt = t_long[1] - t_long[0]
        keys = list(model.dam_dict.keys())
        self.dam_emp = {key: {} for key in keys}
        dam_long = {key: None for key in keys} # dict of arrays for integration
        print("Labelling long traj. dt = {}, Nt = {}, tmin = {}, tmax = {}".format(dt,Nt,t_long[0],t_long[-1]))
        self.long_from_label = np.zeros(len(t_long), dtype=int) # nan is indeterminate; -1 is A; 1 is B
        self.long_to_label = np.zeros(len(t_long), dtype=int)
        for k in range(len(keys)):
            self.dam_emp[keys[k]]['Dc_x'] = np.nan*np.ones(len(t_long))
            self.dam_emp[keys[k]]['x_Dc'] = np.nan*np.ones(len(t_long))
            self.dam_emp[keys[k]]['ab'] = []
            self.dam_emp[keys[k]]['ba'] = []
            dam_long[keys[k]] = model.dam_dict[keys[k]]['pay'](x_long)
        ina_long = (model.adist(x_long)==0)
        #print("ina_long.shape={}".format(ina_long.shape))
        inb_long = (model.bdist(x_long)==0)
        print("sum(ina_long) = {}".format(np.sum(ina_long)))
        print("sum(inb_long) = {}".format(np.sum(inb_long)))
        self.long_from_label[0] = 1*inb_long[0] - 1*ina_long[0]
        self.long_to_label[-1] = 1*inb_long[-1] - 1*ina_long[-1]
        case = np.nan
        for ti in range(1,Nt):
            # Logic for committor
            if ina_long[ti]:
                self.long_from_label[ti] = -1
            elif inb_long[ti]:
                self.long_from_label[ti] = 1
            else:
                self.long_from_label[ti] = self.long_from_label[ti-1]
            if ina_long[-ti]:
                self.long_to_label[-ti] = -1
            elif inb_long[-ti]:
                self.long_to_label[-ti] = 1
            else:
                self.long_to_label[-ti] = self.long_to_label[-ti+1]
            # Logic for integrated damage
            if ina_long[ti] or inb_long[ti]:
                for k in range(len(keys)):
                    self.dam_emp[keys[k]]['Dc_x'][ti] = 0.0
                #tau_a_bwd[ti] = 0.0
            else:
                for k in range(len(keys)):
                    self.dam_emp[keys[k]]['Dc_x'][ti] = self.dam_emp[keys[k]]['Dc_x'][ti-1] + 0.5*dt*(dam_long[keys[k]][ti-1] + dam_long[keys[k]][ti])
                #tau_a_bwd[ti] = tau_a_bwd[ti-1] + dt
            if ina_long[-ti] or inb_long[-ti]:
                for k in range(len(keys)):
                    self.dam_emp[keys[k]]['x_Dc'][-ti] = 0.0
            else:
                for k in range(len(keys)):
                    self.dam_emp[keys[k]]['x_Dc'][-ti] = self.dam_emp[keys[k]]['x_Dc'][-ti+1] + 0.5*dt*(dam_long[keys[k]][-ti+1] + dam_long[keys[k]][-ti])
        ab_reactive_flag = 1*(self.long_from_label==-1)*(self.long_to_label==1)
        ba_reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==-1)
        num_rxn = np.sum(np.diff(ab_reactive_flag)==1)
        num_rxn = min(num_rxn,np.sum(np.diff(ba_reactive_flag)==1))
        print("num_rxn = {}".format(num_rxn))
        ab_starts = np.where(np.diff(ab_reactive_flag)==1)[0] + 1
        ab_ends = np.where(np.diff(ab_reactive_flag)==-1)[0] + 1
        ba_starts = np.where(np.diff(ba_reactive_flag)==1)[0] + 1
        ba_ends = np.where(np.diff(ba_reactive_flag)==-1)[0] + 1
        # Collect the integral distributions
        for k in range(len(keys)):
            for ri in range(num_rxn):
                self.dam_emp[keys[k]]['ab'] += [0.5*dt*(dam_long[keys[k]][ab_starts[ri]] + 2*np.sum(dam_long[keys[k]][ab_starts[ri]:ab_ends[ri]]) + dam_long[keys[k]][ab_ends[ri]])]
                self.dam_emp[keys[k]]['ba'] += [0.5*dt*(dam_long[keys[k]][ba_starts[ri]] + 2*np.sum(dam_long[keys[k]][ba_starts[ri]:ba_ends[ri]]) + dam_long[keys[k]][ba_ends[ri]])]
            self.dam_emp[keys[k]]['ab'] = np.array(self.dam_emp[keys[k]]['ab'])
            self.dam_emp[keys[k]]['ba'] = np.array(self.dam_emp[keys[k]]['ba'])
        # ------------ Compute the empirical rate and its variance -------------
        self.ab_starts = ab_starts
        self.ab_ends = ab_ends
        self.ba_starts = ba_starts
        self.ba_ends = ba_ends
        self.emp_rate = len(ab_starts)/(t_long[-1] - t_long[0])
        num_blocks = 10
        block_size = int((len(t_long)-1)/num_blocks)
        Nt = num_blocks*block_size
        emp_rate_blocks = np.zeros(num_blocks)
        for i in range(num_blocks):
            istart = i*block_size
            iend = (i+1)*block_size
            emp_rate_blocks[i] = np.sum((self.ab_starts >= istart)*(self.ab_starts < iend))/(t_long[iend] - t_long[istart])
        self.emp_rate_unc = np.std(emp_rate_blocks)
        self.emp_return_time_unc = np.std(1/emp_rate_blocks)
        return ab_starts,ab_ends,ba_starts,ba_ends,self.dam_emp
    def compile_data(self,model,istart=0):
        print("In TPT: self.nshort = {}".format(self.nshort))
        t_short,x_short = model.load_short_traj(self.short_simfolder,self.nshort,istart=istart)
        data = Data(x_short,t_short,self.lag_time_seq)
        self.aidx = np.where(model.adist(data.X[:,0])==0)[0]
        self.bidx = np.where(model.bdist(data.X[:,0])==0)[0]
        return data
    def regress_leadtime_modular(self,model,data,theta_fun,method='LASSO'):
        # Given an observable subspace, do a sparse linear regression to estimate the lead time
        eps = 1e-2
        keys = list(model.dam_dict.keys())
        weights = self.chom
        comm_fwd = self.dam_moments['one']['xb'][0,:,0]
        tb = self.dam_moments['one']['xb'][1,:,0]*(comm_fwd > eps)/(comm_fwd + 1*(comm_fwd < eps))
        tb[np.where(comm_fwd <= 1e-2)[0]] = np.nan
        #comm_bwd = self.dam_moments[keys[0]]['ax'][0,:,0]
        midrange_idx = np.where((comm_fwd>0.2)*(comm_fwd<0.8)*(np.isnan(tb)==0))[0]
        #weights = (self.chom*comm_fwd*comm_bwd)[midrange_idx]
        weights = self.chom[midrange_idx]*(comm_fwd[midrange_idx]*(1-comm_fwd[midrange_idx]))**0
        weights *= 1.0/np.sum(weights)
        idx = np.random.choice(midrange_idx,size=min(100000,len(midrange_idx)),p=weights,replace=False)
        log_time = np.log(tb[idx])
        theta_x = theta_fun(data.X[idx,0])
        if method == 'LASSO':
            clf = linear_model.Lasso(alpha=0.08)
            clf.fit(theta_x,log_time)
            recon = clf.predict(theta_x)
            beta = clf.coef_
            intercept = clf.intercept_
            score = clf.score(theta_x,log_time)
        return beta,intercept,score,recon
    def regress_committor_modular(self,model,data,theta_fun,method='LASSO'):
        # Given an observable subspace, do a sparse linear regression to estimate the committor
        keys = list(model.dam_dict.keys())
        weights = self.chom
        comm_fwd = self.dam_moments[keys[0]]['xb'][0,:,0]
        #comm_bwd = self.dam_moments[keys[0]]['ax'][0,:,0]
        midrange_idx = np.where((comm_fwd>0.2)*(comm_fwd<0.8))[0]
        #weights = (self.chom*comm_fwd*comm_bwd)[midrange_idx]
        weights = self.chom[midrange_idx]*(comm_fwd[midrange_idx]*(1-comm_fwd[midrange_idx]))**0
        weights *= 1.0/np.sum(weights)
        idx = np.random.choice(midrange_idx,size=min(100000,len(midrange_idx)),p=weights,replace=False)
        logit_committor = np.log(comm_fwd[idx]/(1-comm_fwd[idx]))
        theta_x = theta_fun(data.X[idx,0])
        if method == 'OMP':
            omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=50)
            omp.fit(theta_x,logit_committor)
            beta = omp.coef_
            intercept = clf.intercept_
            recon = omp.predict(theta_x)
            score = omp.score(theta_x,logit_committor)
        if method == 'LASSO':
            clf = linear_model.Lasso(alpha=0.08)
            clf.fit(theta_x,logit_committor)
            recon = clf.predict(theta_x)
            beta = clf.coef_
            intercept = clf.intercept_
            score = clf.score(theta_x,logit_committor)
        if method == 'OLS':
            lm = linear_model.LinearRegression()
            lm.fit(theta_x,logit_committor)
            recon = lm.predict(theta_x)
            beta = lm.coef_
            intercept = clf.intercept_
            score = lm.score(theta_x,logit_committor)
        return beta,intercept,score,recon
    def lasso_predict_allz(self,model,X):
        # TODO
        theta_x = model.regression_features(X)
        pred = self.intercept_allz + theta_x.dot(self.beta_allz)
        pred = 1.0/(1.0 + np.exp(-pred))
        return pred
    def lasso_predict_onez(self,model,X):
        # TODO
        q = model.q
        n = q['Nz'] - 1
        #zi = np.argmin(np.abs(z - q['z_d'][1:-1]/1000))
        #print("Inside lasso_predict_onez: z={}, z_d/100 = [{}...{}], zi = {}".format(z,q['z_d'][1]/1000,q['z_d'][-2]/1000,zi))
        pred = np.zeros((len(X),n))
        theta_x = model.regression_features(X)
        for zi in range(n):
            theta_x_z = theta_x[:,[j*n+zi for j in range(theta_x.shape[1]//n)]] 
            pred[:,zi] = self.intercept_onez[zi] + theta_x_z.dot(self.beta_onez[zi])
        pred = 1.0/(1.0 + np.exp(-pred))
        return pred
    def plot_transition_ensemble_multiple(self,model,data,theta1d_list,trans_id):
        # VESTIGIAL
        trans_card = np.array([trans_id])
        # TODO
        q = model.q
        # Plot several fields to correlate with committor
        funlib = model.observable_function_library()
        comm_fwd = self.dam_moments['one']['xb'][0]
        #theta1d_list = ['U','mag','vT','LASSO']
        #theta1d_list = ['Uref','vTref']
        n = q['Nz']-1
        figheight = 20*(len(theta1d_list)+1)
        fig,ax = plt.subplots(ncols=1,nrows=2+len(theta1d_list),figsize=(48,figheight),sharex=False,constrained_layout=True)
        # Conditional time to B next
        mfpt = self.dam_moments['one']['xb'][1,:,0].copy()
        mfpt *= 1*(comm_fwd[:,0]>1e-1)/(comm_fwd[:,0] + 1*(comm_fwd[:,0]<=1e-1))
        mfpt[np.where(comm_fwd[:,0]<=1e-1)[0]] = np.nan
        print("MFPT_b: min={}, max={}, mean={}, nan frac = {}".format(np.nanmin(mfpt),np.nanmax(mfpt),np.nanmean(mfpt),np.mean(np.isnan(mfpt))))
        qp0idx = np.where(comm_fwd[:,0]<.1)[0]
        print("MFPT_b where q+ < 0.1: min={}, max={}, mean={}, nan frac = {}".format(np.nanmin(mfpt[qp0idx]),np.nanmax(mfpt[qp0idx]),np.nanmean(mfpt[qp0idx]),np.mean(np.isnan(mfpt[qp0idx]))))
        #sys.exit()
        _,_ = self.plot_transition_ensemble(model,data,mfpt,trans_card,r"$E[\tau^+|x\to B]$","mfpt_xb",committor_flag=False,fig=fig,ax=ax[0],display_lead=False,display_qhalf=False,display_ab=False)
        ax[1].set_xlabel("Time before SSW (days)",fontdict=ggiantfont)
        print("Finished the MFPT part!")
        # Committor first
        _,_ = self.plot_transition_ensemble(model,data,comm_fwd[:,0],trans_card,r"$q^+$","comm_fwd",committor_flag=True,fig=fig,ax=ax[1],display_lead=False,display_ab=False)
        ax[0].set_xlabel("Time before SSW (days)",fontdict=ggiantfont)
        # Then all the other fields
        for i in range(len(theta1d_list)):
            print("Transition ensemble {}".format(theta1d_list[i]))
            if theta1d_list[i] == 'LASSO':
                fieldname = 'LASSO'
                units = 1.0
                unit_symbol = ""
                field_fun = None
                def tfl(x):
                    return self.lasso_predict_allz(model,x).flatten()
                field_fun = tfl
                field = self.lasso_predict_allz(model,data.X[:,0]).flatten()
            else:
                fieldname = funlib[theta1d_list[i]]["name"] #+ ' ({} km)'.format(model.ref_alt)
                units = funlib[theta1d_list[i]]["units"]
                unit_symbol = funlib[theta1d_list[i]]["unit_symbol"]
                #field_fun = model.fun_at_level(funlib[theta1d_list[i]]["fun"],model.ref_alt)
                field_fun = funlib[theta1d_list[i]]["fun"]
                field = field_fun(data.X[:,0]).flatten()
                print("fieldname = {}".format(fieldname))
            _,_ = self.plot_transition_ensemble(model,data,field,trans_card,fieldname,theta1d_list[i],fig=fig,ax=ax[i+2],units=units,unit_symbol=unit_symbol,field_fun=field_fun,display_ab=False)
            ax[i+2].set_xlabel("Time before SSW (days)",fontdict=ggiantfont)
        fig.savefig(join(self.savefolder,"transition_ensemble_tid{}".format(trans_id)))
        plt.close(fig)
        print("Saving new transition ensemble")
        return
    def plot_transition_ensemble(self,model,data,field,trans_card,fieldname,field_abb,committor_flag=False,fig=None,ax=None,units=1.0,unit_symbol="",field_fun=None,display_lead=True,display_qhalf=True,display_ab=True):
        # VESTIGIAL
        # TODO
        q = model.q
        # Calculate the committor-half equivalent level for the field by looking at the field as a function of the committor. 
        # Old:
        comm_fwd = self.dam_moments['one']['xb'][0]
        idx = np.where(np.abs(comm_fwd[:,0] - 0.5) < 0.025)
        inverse_field_qhalf_level = np.sum(field[idx]*self.chom[idx])/np.sum(self.chom[idx])
        print("field.shape = {}".format(field.shape))
        print("field summary: {}".format(describe(field)))
        print("For field {}, inverse qhalf level = {}".format(fieldname,inverse_field_qhalf_level))
        # New:
        qlevels = np.array([0.25,0.5,0.75])
        field_qlevels,qab_levels = self.inverse_committor_slice(field,qlevels)
        field_qhalf_level = field_qlevels[1]
        relationship_sign = np.sign(field_qlevels[-1] - field_qlevels[0])
        print("fieldname: {}".format(fieldname))
        print("range: {},{}".format(np.min(field)*units,np.max(field)*units))
        print("field_qhalf_level: {}".format(field_qhalf_level*units))
        print("Inverse field_qhalf_level: {}".format(inverse_field_qhalf_level*units))
        print("Ratio E[(q+^2)(q-)]/zeta^2 = {}".format(qab_levels/qlevels))
        print("Degree of false negative: E[q|th=th_(1/2), A->B] - E[q|th=th_(1/2)] = {}".format(qab_levels - qlevels))
        #sys.exit()
        # Also save the MFPT averaged on the field_qhalf_level surface
        idx = np.where(np.abs((field - field_qhalf_level)/field_qhalf_level) < 0.05)[0]
        print("How many close to qhalf? {}".format(len(idx)))
        if len(idx) == 0: sys.exit("Nothing close to qhalf level")
        #mfpt_qhalf = np.sum(self.mfpt_b_ava_fwd[idx,0]*self.chom[idx])/np.sum(self.chom[idx])
        #print("mfpt_qhalf: {}".format(mfpt_qhalf))

        # Plot both the field of choice and the committor
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        ab_reactive_flag = 1*(self.long_from_label==-1)*(self.long_to_label==1)
        ba_reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==-1)
        # What is the mean transit time?
        ab_starts = np.where(np.diff(ab_reactive_flag)==1)[0] + 1
        ab_ends = np.where(np.diff(ab_reactive_flag)==-1)[0] + 1
        ba_starts = np.where(np.diff(ba_reactive_flag)==1)[0] + 1
        ba_ends = np.where(np.diff(ba_reactive_flag)==-1)[0] + 1
        # Get them lined up correctly
        if ab_starts[0] > ab_ends[0]: ab_ends = ab_ends[1:]
        if ab_starts[-1] > ab_ends[-1]: ab_starts = ab_starts[:-1]
        if ba_starts[0] > ba_ends[0]: ba_ends = ba_ends[1:]
        if ba_starts[-1] > ba_ends[-1]: ba_starts = ba_starts[:-1]
        # --------------------------
        # Compute all strip widths
        numstrips = min(len(ab_starts),100)
        ab_ss = np.arange(numstrips)
        #ab_ss = np.random.choice(np.arange(len(ab_starts)),size=numstrips,replace=False)
        strip_widths = np.zeros(numstrips)
        for i in range(numstrips):
            idx0,idx1= ab_starts[ab_ss[i]],ab_ends[ab_ss[i]]
            if field_fun is None:
                field_long = self.out_of_sample_extension(field,data,x_long[idx0:idx1])
            else:
                field_long = field_fun(x_long[idx0:idx1]).flatten()
            last_qhalf_crossing_idx = idx0 + np.where((field_long-field_qhalf_level)*(field_long[-1]-field_qhalf_level) < 0)[0]
            if len(last_qhalf_crossing_idx) == 0:
                last_qhalf_crossing_projected = t_long[idx0]
            else:
                last_qhalf_crossing_projected = t_long[last_qhalf_crossing_idx[-1] + 1]
            comm_long = self.out_of_sample_extension(comm_fwd[:,0],data,x_long[idx0:idx1])
            last_qhalf_crossing = t_long[idx0 + np.where(comm_long <= 0.5)[0][-1] + 1]
            print("\tTransition number {}: Timespan = {}, last_qhalf_crossing = {}, last_qhalf_crossing_projected = {}".format(i,t_long[idx1]-t_long[idx0],last_qhalf_crossing-t_long[idx0],last_qhalf_crossing_projected-t_long[idx0]))
            strip_widths[i] = last_qhalf_crossing_projected - last_qhalf_crossing
        print("For field {}, mean(strip_widths) = {}".format(fieldname,np.mean(strip_widths)))
        # --------------------------

        # AB transitions
        #fig,ax = plt.subplots(nrows=2,figsize=(45,15))
        n = q['Nz']-1
        zi = q['zi']
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(48,12))
        colorlist = ['black'] #['lawngreen','purple'] #,'black'] #,'firebrick','cyan']
        thalf_list = [[] for _ in range(len(colorlist))]
        thalf_list_projected = [[] for _ in range(len(colorlist))]
        num_trans = min(len(colorlist),len(ab_starts))
        #trans_card = np.array([trans_card]) #[0,2]
        min_start = 0.0
        max_end = 0.0
        ab_long = units*self.out_of_sample_extension(field,data,model.tpt_obs_xst)
        ymin = np.min(units*field_qhalf_level)
        ymax = np.max(units*field_qhalf_level)
        print("After field_qhalf_level: ymin={}, ymax={}".format(ymin,ymax))
        if display_ab:
            ymin = min(ymin,np.min(ab_long))
            ymax = max(ymax,np.min(ab_long))
        print("After ab_long: ymin={}, ymax={}".format(ymin,ymax))
        # First set the bounds in the plot
        for i in range(num_trans):
            idx0,idx1= ab_starts[trans_card[i]],ab_ends[trans_card[i]]
            b2_idx = np.where(model.bdist(x_long[idx0:idx1]) == 0)[0]
            #b2_idx = np.where(x_long[idx0:idx1,2*n+zi] <= self.bcv_compute[2*n+zi])[0]
            #print("len(b2_idx) = {}".format(len(b2_idx)))
            if False: #len(b2_idx) > 0:
                t_offset = t_long[idx0+b2_idx[0]]  # new
            else:
                t_offset = t_long[idx1]
            #print("first t = {}".format(t_long[idx0] - t_offset))
            min_start = min(min_start,t_long[idx0]-t_offset)
            max_end = max(max_end,t_long[idx1]-t_offset)
        radius = max(max_end,-min_start)
        min_start = -radius
        max_end = 0.25*radius
        for i in range(num_trans):
            idx0,idx1= ab_starts[trans_card[i]],ab_ends[trans_card[i]]
            b2_idx = np.where(model.bdist(x_long[idx0:idx1]) == 0)[0]
            #b2_idx = np.where(x_long[idx0:idx1,2*n+zi] <= self.bcv_compute[2*n+zi])[0]
            print("len(b2_idx) = {}".format(len(b2_idx)))
            if len(b2_idx) > 0:
                t_offset = t_long[idx0+b2_idx[0]]  # new
            else:
                t_offset = t_long[idx1]
            #print("first t = {}".format(t_long[idx0] - t_offset))
            idx0_early = np.argmin(np.abs(t_offset + min_start - t_long))
            idx1_late = np.argmin(np.abs(t_offset + max_end - t_long))
            if field_fun is None:
                field_long = self.out_of_sample_extension(field,data,x_long[idx0_early:idx1_late])
            else:
                field_long = field_fun(x_long[idx0_early:idx1_late]).flatten()
            ymin = min(ymin,np.min(field_long)*units)
            ymax = max(ymax,np.max(field_long)*units)
            comm_long = self.out_of_sample_extension(comm_fwd[:,0],data,x_long[idx0_early:idx1_late])
            print("comm_long = {}...{}".format(comm_long[:4],comm_long[-4:]))
            # Re-align the offsets at the time when U first dips negative
            ax.plot(t_long[idx0_early:idx1_late]-t_offset,field_long*units,color=colorlist[i],linewidth=12)
            print("For field {}, range(field_long*units) = {},{}".format(fieldname,np.min(field_long*units),np.max(field_long*units)))
            #sys.exit()
            #ax[1].plot(t_long[idx0:idx1]-t_long[idx0],comm_long,color=colorlist[i],linewidth=2)
            # --------------------------------------------------
            # This is the gd impossible part: delineate strips
            half_crossings = np.where(np.diff(np.sign(comm_long-0.5)) == 2)[0]
            print("len(half_crossings) = {}".format(len(half_crossings)))
            print("field_qhalf_level = {}".format(field_qhalf_level*units))
            half_crossings_projected = np.where((np.abs(np.diff(np.sign(field_long-field_qhalf_level))) == 2)*(t_long[idx0_early+1:idx1_late] < t_offset))[0]
            if len(half_crossings_projected) > 0 and display_qhalf:
                ax.plot((t_long[idx0 + half_crossings_projected[-1]] - t_offset)*np.ones(2),ax.get_ylim(),color='black',linewidth=12,linestyle='--')
            #if len(half_crossings_projected) == 0:
            #    sys.exit("No half crossings")
            #thalf_list[i] = np.zeros(len(half_crossings))
            #thalf_list_projected[i] = np.zeros(len(half_crossings_projected))
            #print("len(half_crossings) = {}".format(len(half_crossings)))
            #for j in range(len(half_crossings)):
            #    if committor_flag:
            #        vert = 0.5
            #    else:
            #        vert = (field_long[half_crossings[j]]+field_long[half_crossings[j]+1])/2
            #    thalf_list[i][j] = t_long[idx0_early+half_crossings[j]]-t_offset
            #for j in range(len(half_crossings_projected)):
            #    if committor_flag:
            #        vert = 0.5
            #    else:
            #        vert = (field_long[half_crossings_projected[j]]+field_long[half_crossings_projected[j]+1])/2
            #    thalf_list_projected[i][j] = t_long[idx0_early+half_crossings_projected[j]]-t_offset
            #    #ax.scatter(t_long[idx0_early+half_crossings[j]]-t_offset,units*vert,marker='o',color=colorlist[i],s=6000,linewidth=8)
            ax.plot(np.zeros(2),ax.get_ylim(),color='black',linewidth=12)
        ylim_margin = 0.02*(ymax - ymin)
        ax.set_ylim([ymin-ylim_margin,ymax+ylim_margin])
        #for i in range(num_trans):
        #    #for j in range(len(thalf_list[i])):
        #    #    #ax.plot(thalf_list[i][j]*np.ones(2),ax.get_ylim(),color=colorlist[i],linestyle='--',linewidth=10)
        #    #    ax.axvspan(thalf_list[i][j],thalf_list_projected[i][j],facecolor='gray',alpha=0.25,zorder=-1)
        #    ax.axvspan(thalf_list[i][-1],thalf_list_projected[i][-1],facecolor='deepskyblue',alpha=0.25,zorder=-1)
        # ---------------------------------------------------
        if display_ab:
            ax.plot(np.array([min_start,max_end]),ab_long[0]*np.ones(2),color='skyblue',linewidth=10.0)
            ax.plot(np.array([min_start,max_end]),ab_long[1]*np.ones(2),color='red',linewidth=10.0)
        if display_qhalf: ax.plot(np.array([min_start,max_end]),field_qhalf_level*units*np.ones(2),color='black',linewidth=15.0,linestyle='--',zorder=10)
        ylab = fieldname
        #if len(unit_symbol) > 0: ylab += " (%s)"%(unit_symbol)
        ax.set_title(ylab,fontdict=ggiantfont)
        if len(unit_symbol) > 0: ax.set_ylabel(unit_symbol, fontdict=ggiantfont)
        ax.tick_params(axis='x',labelsize=100)
        ax.tick_params(axis='y',labelsize=100)
        # Put onto the plot the average and standard deviation of lead time
        mean_lead = np.mean(strip_widths)
        std_lead = np.std(strip_widths)
        if display_lead: ax.text(min_start,np.mean(ax.get_ylim()),r"$q^+$ lead $=%.1f\pm%.1f$"%(mean_lead,std_lead),fontdict=giantfont,bbox=dict(boxstyle='round',facecolor='wheat'))

        #ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        #ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
        #ax.locator_params(tight=True,nbins=3,axis='y')
        #ab_long = self.out_of_sample_extension(self.comm_fwd,np.array([self.acv_compute,self.bcv_compute]))
        #ax[1].set_xlabel("Time (days)",fontdict=bigfont)
        #ax[1].set_ylabel("Committor",fontdict=bigfont)
        #ax[0].set_title("Long integration",fontdict=font)
        #ax[1].tick_params(axis='x',labelsize=40)
        #ax[1].tick_params(axis='y',labelsize=40)
        #fig.savefig(join(self.savefolder,"{}_ensemble_ab".format(field_abb)))
        return fig,ax
    def plot_field_long(self,model,data,field,fieldname,field_abb,field_fun=None,units=1.0,tmax=70,field_unit_symbol=None,time_unit_symbol=None,phases=['aa','ab','ba','bb'],density_1d_flag=True):
        print("Beginning plot field long")
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        #self.display_1d_densities_emp(model,data,[field_abb],'vertial',phases=phases,save_flag=True)
        tmax = min(tmax,t_long[-1])
        ab_reactive_flag = 1.0*(self.long_from_label==-1)*(self.long_to_label==1)
        ba_reactive_flag = 1.0*(self.long_from_label==1)*(self.long_to_label==-1)
        #sys.exit("sum(ab_reactive_flag) = {}".format(np.sum(ab_reactive_flag)))
        any_trans = (np.sum(ab_reactive_flag) > 0) and (np.sum(ba_reactive_flag) > 0)
        if any_trans:
            # Identify the transitions
            ab_starts = np.where(np.diff(ab_reactive_flag)==1)[0] + 1
            ab_ends = np.where(np.diff(ab_reactive_flag)==-1)[0] + 1
            ba_starts = np.where(np.diff(ba_reactive_flag)==1)[0] + 1
            ba_ends = np.where(np.diff(ba_reactive_flag)==-1)[0] + 1
            # Get them lined up correctly
            if ab_starts[0] > ab_ends[0]: ab_ends = ab_ends[1:]
            if ab_starts[-1] > ab_ends[-1]: ab_starts = ab_starts[:-1]
            if ba_starts[0] > ba_ends[0]: ba_ends = ba_ends[1:]
            if ba_starts[-1] > ba_ends[-1]: ba_starts = ba_starts[:-1]
        print(f"num_trans = {len(ab_starts)}")
        # Plot, marking transitions
        tmax = min(t_long[-1],tmax)
        # Interpolate field
        timax = np.argmin(np.abs(t_long-tmax))
        tsubset = np.linspace(0,timax-1,min(timax,5000)).astype(int)
        if field_fun is None:
            field_long = self.out_of_sample_extension(field,data,x_long[tsubset])
            if field_abb == 'qp':
                ab_long = np.array([0,1])
            else:
                ab_long = self.out_of_sample_extension(field,data,model.tpt_obs_xst)
        else:
            field_long = field_fun(x_long[tsubset]).flatten()
            ab_long = field_fun(model.tpt_obs_xst).flatten()
        print("field_long.shape = {}".format(field_long.shape))
        fig,ax = plt.subplots(ncols=2,figsize=(22,7),gridspec_kw={'width_ratios': [3,1]},sharey=True)
        ax[0].plot(t_long[tsubset],units*field_long,color='black')
        ax[0].plot(t_long[[tsubset[0],tsubset[-1]]],ab_long[0]*np.ones(2)*units,color='skyblue',linewidth=2.5)
        ax[0].plot(t_long[[tsubset[0],tsubset[-1]]],ab_long[1]*np.ones(2)*units,color='red',linewidth=2.5)
        dthab = np.abs(ab_long[1]-ab_long[0])
        #ax[0].text(0,units*(ab_long[0]+0.01*dthab),asymb,fontdict=bbigfont,color='black',weight='bold')
        #ax[0].text(0,units*(ab_long[1]+0.01*dthab),bsymb,fontdict=bbigfont,color='black',weight='bold')
        if any_trans:
            for i in range(len(ab_starts)):
                if ab_ends[i] < timax:
                    ax[0].axvspan(t_long[ab_starts[i]],t_long[ab_ends[i]],facecolor='orange',alpha=0.5,zorder=-1)
            for i in range(len(ba_starts)):
                if ba_ends[i] < timax:
                    ax[0].axvspan(t_long[ba_starts[i]],t_long[ba_ends[i]],facecolor='mediumspringgreen',alpha=0.5,zorder=-1)
        xlab = "Time"
        if time_unit_symbol is not None: xlab += " [{}]".format(time_unit_symbol)
        ax[0].set_xlabel(xlab,fontdict=bigfont)
        ylab = fieldname
        if field_unit_symbol is not None: ylab += " [{}]".format(field_unit_symbol)
        ax[0].set_ylabel(ylab,fontdict=bigfont)
        ylim = ax[0].get_ylim()
        fmt_y = helper.generate_sci_fmt(ylim[0],ylim[1])
        ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
        #ax.set_title("Long integration",fontdict=font)
        ax[0].tick_params(axis='both',labelsize=25)
        #ax.yaxis.set_major_locator(ticker.NullLocator())
        # Now plot the densities in y
        if density_1d_flag:
            self.display_1d_densities(model,data,[field_abb],'vertical',fig=fig,ax=ax[1],phases=phases)
            ax[1].yaxis.set_visible(False)
            ax[1].set_xlabel("Probability density",fontdict=bigfont)
            ax[1].tick_params(axis='both',labelsize=25)
        fig.savefig(join(self.savefolder,"{}_long".format(field_abb)),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        print("Done plotting field long")
        # ------------ Next: plot only some transitions on their own plot --------------
        quantiles = np.array([0.15,0.25,0.4,0.5])
        if any_trans:
            avg_prea_flag = True
            fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(12,12),sharex=True,sharey=True)
            num_trans = min(300,len(ab_starts))
            print(f"num_trans = {num_trans}")
            trans_colors = ['red','cyan','black']
            num_colored_trans = len(trans_colors)
            # Select them from across the distribution of transit times
            transit_quantiles = np.linspace(0,1,num_colored_trans+2)[1:-1]
            colored_idx = np.zeros(num_colored_trans, dtype=int)
            transit_distribution = np.array([ab_ends[i] - ab_starts[i] for i in range(num_trans)])
            for i in range(num_colored_trans):
                transit_time_quantile = np.quantile(transit_distribution,transit_quantiles[i])
                colored_idx[i] = np.argmin(np.abs(transit_distribution - transit_time_quantile))
            #max_duration = max([ab_ends[i]-ab_starts[i] for i in range(num_trans)])
            # Match the maximum duration with the later plots of flux distribution
            max_duration = max(int(90/(t_long[1]-t_long[0])), max([ab_ends[i]-ab_starts[i] for i in colored_idx]) + 1)
            print("max_duration = {}".format(max_duration))
            field_trans_composite = np.zeros((num_trans,max_duration))
            for i in range(num_trans):
                if i % 10 == 0: print(f"Plotting transition {i} out of {num_trans}")
                k0,k1 = ab_starts[i]-1,ab_ends[i]
                k0_padded = k1 - max_duration # Get all the beginnings lined up
                if field_fun is None:
                    field_trans = self.out_of_sample_extension(field,data,x_long[k0_padded:k1],k=15)
                else:
                    field_trans = field_fun(x_long[k0_padded:k1]).flatten()
                if (not avg_prea_flag) and (k0 > k0_padded+1):
                    field_trans[:(k0-k0_padded-1)] = np.nan
                ax[0,0].plot(t_long[k0_padded:k1]-t_long[k1],field_trans*units,color='gray',alpha=0.65,linewidth=0.75,zorder=-1)
                if i in colored_idx:
                    color = trans_colors[np.where(colored_idx==i)[0][0]]
                    alpha = 1.0
                    linewidth = 2
                    ax[0,0].plot(t_long[k0:k1]-t_long[k1],field_trans[-(k1-k0):]*units,color=color,alpha=1.0,linewidth=2,zorder=1)
                field_trans_composite[i] = field_trans
                #pad = max_duration - (k1-k0)
                #print("pad = ", pad)
                #if field_fun is None:
                #    field_trans = self.out_of_sample_extension(field,data,x_long[k0-pad:k1+pad])
                #else:
                #    field_trans = field_fun(x_long[k0-pad:k1+pad]).flatten()
                #print("field_trans.shape = {}".format(field_trans.shape))
                #ax[0].plot(t_long[k0-pad:k1]-t_long[k1],field_trans[:k1-k0+pad]*units,color='gray',alpha=0.25,linewidth=1,zorder=-1)
                #if i in colored_idx:
                #    color = trans_colors[np.where(colored_idx==i)[0][0]]
                #    alpha = 1.0
                #    linewidth = 2
                #    ax[0].plot(t_long[k0:k1]-t_long[k1],field_trans[pad:k1-k0+pad]*units,color=color,alpha=1.0,linewidth=2,zorder=1)
                #field_trans_composite[i] = field_trans[:k1-k0+pad]
            # Plot the quantiles
            for qi in range(len(quantiles)):
                lower = np.nanquantile(field_trans_composite, quantiles[qi], axis=0)
                if qi < len(quantiles)-1:
                    upper = np.nanquantile(field_trans_composite, 1-quantiles[qi], axis=0)
                    reds = (qi+1)/len(quantiles)
                    ax[1,0].fill_between(t_long[:max_duration]-t_long[max_duration],lower*units,upper*units,color=plt.cm.Reds(reds),alpha=1.0,zorder=qi)
                else:
                    ax[1,0].plot(t_long[:max_duration]-t_long[max_duration],lower*units,color='black',zorder=qi+10,linestyle='-',linewidth=2)
            xlab = r"Time$-\tau_B^+$"
            ylab = fieldname
            if field_unit_symbol is not None: ylab += " [{}]".format(field_unit_symbol)
            for i in range(2):
                ax[i,0].axhline(ab_long[0]*units,color='skyblue',zorder=-1)
                ax[i,0].axhline(ab_long[1]*units,color='red',zorder=-1)
                ax[i,0].set_ylabel(ylab,fontdict=font)
                ylim = ax[i,0].get_ylim()
                fmt_y = helper.generate_sci_fmt(ylim[0],ylim[1])
                ax[i,0].yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
            if time_unit_symbol is not None: xlab += " [{}]".format(time_unit_symbol)
            ax[1,0].set_xlabel(xlab,fontdict=font)
            ax[1,0].set_xlim([-max_duration*(t_long[1]-t_long[0]),0])
            # TODO: now plot the second column the same quantity vs. lead time and vs. committor
            fig.savefig(join(self.savefolder,"transitory_{}_nt{}".format(field_abb,num_trans)),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # Save the composite in order to use later for comparison with the TPT composite
            np.save(join(self.savefolder,"composite_{}".format(field_abb)),field_trans_composite)
            np.save(join(self.savefolder,"composite_time"),t_long[:max_duration]-t_long[max_duration])
        del x_long
        return
    def plot_trans_2d_driver(self,model,data):
        num_trans = 100
        # Plot the right combinations of committor, lead time, Uref, and vTintref
        qp = self.dam_moments['one']['xb'][0,:,0]
        tb = self.dam_moments['one']['xb'][1,:,0]
        eps = 0.2
        tb *= (qp > eps)/(qp + 1.0*(qp <= eps))
        tb[qp <= eps] = np.nan
        qp[qp <= eps] = np.nan
        funlib = model.observable_function_library()
        # Uref vs qp
        field_abbs = ["qp","Uref"]
        field_funs = [None,funlib[field_abbs[1]]["fun"]]
        fields = [qp,None]
        field_units = [1.0,funlib[field_abbs[1]]["units"]]
        field_names = [r"$q_B^+$",funlib[field_abbs[1]]["name"]]
        fig,ax = self.plot_trans_2d(model,data,field_funs,fields,field_names,field_units,field_abbs,num_trans=num_trans)
        fig.savefig(join(self.savefolder,"transitory_{}_vs_{}_nt{}".format(field_abbs[1],field_abbs[0],num_trans)),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # vTintref vs qp
        field_abbs = ["qp","vTintref"]
        field_funs = [None,funlib[field_abbs[1]]["fun"]]
        fields = [qp,None]
        field_units = [1.0,funlib[field_abbs[1]]["units"]]
        field_names = [r"$q_B^+$",funlib[field_abbs[1]]["name"]]
        fig,ax = self.plot_trans_2d(model,data,field_funs,fields,field_names,field_units,field_abbs,num_trans=num_trans)
        fig.savefig(join(self.savefolder,"transitory_{}_vs_{}_nt{}".format(field_abbs[1],field_abbs[0],num_trans)),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # tb vs qp
        field_abbs = ["qp","tb"]
        field_funs = [None,None]
        fields = [qp,-tb]
        field_units = [1.0,1.0]
        field_names = [r"$q_B^+$",r"$-\eta_B^+$"]
        fig,ax = self.plot_trans_2d(model,data,field_funs,fields,field_names,field_units,field_abbs,num_trans=num_trans)
        fig.savefig(join(self.savefolder,"transitory_{}_vs_{}_nt{}".format(field_abbs[1],field_abbs[0],num_trans)),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # vTintref vs tb
        field_abbs = ["tb","vTintref"]
        field_funs = [None,funlib[field_abbs[1]]["fun"]]
        fields = [-tb,None]
        field_units = [1.0,funlib[field_abbs[1]]["units"]]
        field_names = [r"$-\eta_B^+$",funlib[field_abbs[1]]["name"]]
        fig,ax = self.plot_trans_2d(model,data,field_funs,fields,field_names,field_units,field_abbs,num_trans=num_trans)
        fig.savefig(join(self.savefolder,"transitory_{}_vs_{}_nt{}".format(field_abbs[1],field_abbs[0],num_trans)),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # Uref vs tb
        field_abbs = ["tb","Uref"]
        field_funs = [None,funlib[field_abbs[1]]["fun"]]
        fields = [-tb,None]
        field_units = [1.0,funlib[field_abbs[1]]["units"]]
        field_names = [r"$-\eta_B^+$",funlib[field_abbs[1]]["name"]]
        fig,ax = self.plot_trans_2d(model,data,field_funs,fields,field_names,field_units,field_abbs,num_trans=num_trans)
        fig.savefig(join(self.savefolder,"transitory_{}_vs_{}_nt{}".format(field_abbs[1],field_abbs[0],num_trans)),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        return
    def plot_trans_2d(self,model,data,field_funs,fields,field_names,field_units,field_abbs,num_trans=100):
        # For a bunch of transitions (A->B), plot them in a 2D subspace. 
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        #self.display_1d_densities_emp(model,data,[field_abb],'vertial',phases=phases,save_flag=True)
        ab_reactive_flag = 1.0*(self.long_from_label==-1)*(self.long_to_label==1)
        #sys.exit("sum(ab_reactive_flag) = {}".format(np.sum(ab_reactive_flag)))
        any_trans = (np.sum(ab_reactive_flag) > 0)
        if any_trans:
            # Identify the transitions
            ab_starts = np.where(np.diff(ab_reactive_flag)==1)[0] + 1
            ab_ends = np.where(np.diff(ab_reactive_flag)==-1)[0] + 1
            # Get them lined up correctly
            if ab_starts[0] > ab_ends[0]: ab_ends = ab_ends[1:]
            if ab_starts[-1] > ab_ends[-1]: ab_starts = ab_starts[:-1]
        num_trans = min(num_trans,len(ab_starts))
        # Compute the field on each segment
        transitions_concat = np.concatenate(tuple([x_long[ab_starts[i]:ab_ends[i]] for i in range(num_trans)]), axis=0)
        field_trans = np.zeros((len(transitions_concat),2))
        abfield = np.zeros((2,2))
        for i in range(2):
            if field_funs[i] is not None:
                field_trans[:,i] = field_funs[i](transitions_concat)
                abfield[:,i] = field_funs[i](model.tpt_obs_xst).flatten()
            else:
                field_trans[:,i] = self.out_of_sample_extension(fields[i],data,transitions_concat,inverse_lengthscale=0.2,k=20)
                abfield[:,i] = self.out_of_sample_extension(fields[i],data,model.tpt_obs_xst).flatten()
        # Now plot them in the 2D subspace
        fig,ax = plt.subplots(nrows=2,figsize=(6,12),sharex=True,sharey=True)
        trans_colors = ['red','cyan','black']
        num_colored_trans = len(trans_colors)
        colored_idx = np.zeros(num_colored_trans, dtype=int)
        transit_distribution = np.array([ab_ends[i] - ab_starts[i] for i in range(num_trans)])
        transit_quantiles = np.linspace(0,1,num_colored_trans+2)[1:-1]
        for i in range(num_colored_trans):
            transit_time_quantile = np.quantile(transit_distribution,transit_quantiles[i])
            colored_idx[i] = np.argmin(np.abs(transit_distribution - transit_time_quantile))
        # Store all the trajectories
        field_list_0 = []
        field_list_1 = []
        k = 0
        for i in range(num_trans):
            if i % 10 == 0:
                print(f"Transition number {i} out of {num_trans}")
            idx = np.arange(k,k+ab_ends[i]-ab_starts[i])
            field_list_0 += [field_trans[idx,0]]
            field_list_1 += [field_trans[idx,1]]
            ax[0].plot(field_trans[idx,0]*field_units[0],field_trans[idx,1]*field_units[1],color='gray',alpha=0.25,linewidth=1,zorder=-1)
            if i in colored_idx:
                color = trans_colors[np.where(colored_idx==i)[0][0]]
                alpha = 1.0
                linewidth=1.5
                ax[0].plot(field_trans[idx,0]*field_units[0],field_trans[idx,1]*field_units[1],color=color,alpha=alpha,linewidth=linewidth,zorder=1)
            k += len(idx)
        #ax.set_xlabel(field_names[0],fontdict=font)
        ax[0].set_ylabel(field_names[1],fontdict=font)
        # Now plot envelopes in the bottom
        field0_range = np.linspace(np.nanmin(field_trans[:,0]),np.nanmax(field_trans[:,0]),32)[1:-1]
        quantiles = np.array([0.05,0.1,0.25,0.4,0.5])
        distribution = np.zeros((2*len(quantiles)-1,len(field0_range)))
        current = np.array([len(quantiles),len(field0_range)])
        for i in range(len(field0_range)):
            surf_locs = np.array([]) # Crossing locations across the surface
            surf_flux = np.array([]) # Crossing flux across the surface
            for j in range(num_trans):
                # Positive crossings
                idx0 = np.where((field_list_0[j][:-1] < field0_range[i])*(field_list_0[j][1:] > field0_range[i]))[0]
                if len(idx0) > 0:
                    surf_locs = np.concatenate((surf_locs,(field_list_1[j][idx0] + field_list_1[j][idx0+1])/2)) # TODO: make this a proper convex combination
                    surf_flux = np.concatenate((surf_flux, np.ones(len(idx0))))
                # Negative crossings
                idx0 = np.where((field_list_0[j][:-1] > field0_range[i])*(field_list_0[j][1:] < field0_range[i]))[0]
                if len(idx0) > 0:
                    surf_locs = np.concatenate((surf_locs,(field_list_0[j][idx0] + field_list_0[j][idx0+1])/2)) # TODO: make this a proper convex combination
                    surf_flux = np.concatenate((surf_flux, -np.ones(len(idx0))))
            # Now locate the quantiles
            print(f"surf_locs = {surf_locs}")
            print(f"surf_flux = {surf_flux}")
            #surf_locs = np.concatenate(surf_locs)
            #surf_flux = np.concatenate(surf_flux)
            order = np.argsort(surf_locs)
            surf_flux = surf_flux[order]
            surf_locs = surf_locs[order]
            cdf = np.cumsum(surf_flux)
            print(f"cdf = {cdf}")
            for qi in range(len(quantiles)):
                idx = np.where(cdf >= cdf[-1]*quantiles[qi])[0]
                if len(idx) > 0:
                    distribution[qi,i] = surf_locs[idx[0]]
                else:
                    distribution[qi,i] = np.nan
                if qi < len(quantiles)-1:
                    idx = np.where(cdf >= cdf[-1]*(1-quantiles[qi]))[0]
                    if len(idx) > 0:
                        distribution[len(quantiles)-1-qi,i] = surf_locs[idx[0]]
                    else:
                        distribution[qi,i] = np.nan
        # Now plot the quantile envelopes: TODO
        for qi in range(len(quantiles)):
            lower = distribution[qi]
            if qi < len(quantiles)-1:
                upper = distribution[len(quantiles)-1-qi]
                ax[1].fill_between(field0_range*field_units[0],lower*field_units[1],upper*field_units[1],color=plt.cm.Reds((qi+1)/len(quantiles)),alpha=1.0,zorder=qi)
            else:
                ax[1].plot(field0_range*field_units[0],lower*field_units[1],color='black',zorder=qi+10,linestyle='-',linewidth=2)
        ax[1].set_xlabel(field_names[0],fontdict=font)
        ax[1].set_ylabel(field_names[1],fontdict=font)
        return fig,ax
    def plot_field_long_2d(self,model,data,fieldnames,field_funs,field_abbs,field_data=None,units=[1.0,1.0],tmax=70,field_unit_symbols=["",""],orientation=None):
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        tmax = min(tmax,t_long[-1])
        ab_reactive_flag = 1*(self.long_from_label==-1)*(self.long_to_label==1)
        ba_reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==-1)
        # Identify the transitions
        ab_starts = np.where(np.diff(ab_reactive_flag)==1)[0] + 1
        ab_ends = np.where(np.diff(ab_reactive_flag)==-1)[0] + 1
        ba_starts = np.where(np.diff(ba_reactive_flag)==1)[0] + 1
        ba_ends = np.where(np.diff(ba_reactive_flag)==-1)[0] + 1
        # Get them lined up correctly
        any_trans = len(ab_starts) > 0 and len(ba_starts) > 0
        if any_trans:
            if ab_starts[0] > ab_ends[0]: ab_ends = ab_ends[1:]
            if ab_starts[-1] > ab_ends[-1]: ab_starts = ab_starts[:-1]
            if ba_starts[0] > ba_ends[0]: ba_ends = ba_ends[1:]
            if ba_starts[-1] > ba_ends[-1]: ba_starts = ba_starts[:-1]
        timax = np.argmin(np.abs(t_long-tmax))
        print("t_long[timax] = {}".format(t_long[timax]))
        tsubset = np.linspace(0,timax-1,min(timax,15000)).astype(int)
        # Plot the two fields vs. each other, marking transitions
        if field_funs[0] is not None:
            field0 = field_funs[0](x_long[tsubset]).flatten()
            ab0 = field_funs[0](model.tpt_obs_xst).flatten()
        else:
            field0 = self.out_of_sample_extension(field_data[0],data,x_long[tsubset])
            ab0 = self.out_of_sample_extension(field_data[0],data,model.tpt_obs_xst)
        if field_funs[1] is not None:
            field1 = field_funs[1](x_long[tsubset]).flatten()
            ab1 = field_funs[1](model.tpt_obs_xst).flatten()
        else:
            field1 = self.out_of_sample_extension(field_data[1],data,x_long[tsubset])
            ab1 = self.out_of_sample_extension(field_data[0],data,model.tpt_obs_xst)
        fig,ax = plt.subplots(figsize=(6,6))
        ax.plot(field0*units[0],field1*units[1],color='black',zorder=0,linewidth=1.0)
        for i in range(len(ab_starts)):
            if ab_ends[i] < timax:
                tss = np.where((t_long[tsubset]>t_long[ab_starts[i]])*(t_long[tsubset]<t_long[ab_ends[i]]))[0]
                ax.plot(field0[tss]*units[0],field1[tss]*units[1],color='darkorange',linewidth=3,zorder=3)
        for i in range(len(ba_starts)):
            if ba_ends[i] < timax:
                tss = np.where((t_long[tsubset]>t_long[ba_starts[i]])*(t_long[tsubset]<t_long[ba_ends[i]]))[0]
                ax.plot(field0[tss]*units[0],field1[tss]*units[1],color='springgreen',linewidth=3,zorder=2)
        ax.text(ab0[0]*units[0],ab1[0]*units[1],asymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=25,horizontalalignment='center',verticalalignment='center',zorder=10)
        ax.text(ab0[1]*units[0],ab1[1]*units[1],bsymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=25,horizontalalignment='center',verticalalignment='center',zorder=10)
        ax.set_xlabel(r"%s [%s]"%(fieldnames[0],field_unit_symbols[0]),fontdict=ffont)
        ax.set_ylabel(r"%s [%s]"%(fieldnames[1],field_unit_symbols[1]),fontdict=ffont)
        ax.tick_params(axis='both',labelsize=20)
        # Set tick formats
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        fmt_x = helper.generate_sci_fmt(xlim[0],xlim[1])
        fmt_y = helper.generate_sci_fmt(ylim[0],ylim[1])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        fig.savefig(join(self.savefolder,"long_{}_{}".format(field_abbs[0],field_abbs[1])),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        del x_long
        return 
    def plot_committor_fidelity(self,q):
        # VESTIGIAL
        ss = np.random.choice(np.arange(self.nshort),size=min(self.nshort,100000),replace=False)
        emp_comm_fwd = 1.0*(self.short_to_label==1)[ss]
        emp_comm_bwd = 1.0*(self.short_from_label==-1)[ss]
        weights = np.ones(len(ss))/len(ss)
        fig,ax,_ = self.plot_field_1d(emp_comm_fwd,weights,self.comm_fwd[ss,0],shp=[20,],funname=r"DGA $q^+$ (probability)",fieldname=r"Empirical $q^+$ (probability)",label="forward",color='black',linestyle='None')
        ax.plot([0,1],[0,1],color='black',linestyle='--')
        ax.set_title(r"$q^+$ fidelity",fontdict=font)
        fig.savefig(join(self.savefolder,"comm_fwd_fidelity"))
        plt.close(fig)
        fig,ax,_ = self.plot_field_1d(emp_comm_bwd,weights,self.comm_bwd[ss,0],shp=[20,],funname=r"DGA $q^-$ (probability)",fieldname=r"Empirical $q^-$ (probability)",label="forward",color='black',linestyle='None')
        ax.plot([0,1],[0,1],color='black',linestyle='--')
        ax.set_title(r"$q^-$ fidelity",fontdict=font)
        fig.savefig(join(self.savefolder,"comm_bwd_fidelity"))
        plt.close(fig)
        print("Done plotting committor fidelity")
        return
    def tabulate_functions(self,data,theta_list,current_flag=True,ss=None):
        # Tabulate a list of theta functions on the data
        if ss is None: ss = np.arange(self.nshort)
        print("ss range = (%d,%d)"%(np.min(ss),np.max(ss)))
        bndy_dist = lambda x: np.minimum(self.adist(x),self.bdist(x))
        data.insert_boundaries(bndy_dist,lag_time_max=self.lag_time_current)
        Nth = len(theta_list)
        theta_x = np.zeros((len(ss),Nth))
        for i in range(Nth):
            theta_x[:,i] = theta_list[i](data.X[ss,0,:]).flatten()
            ans = theta_x
        if current_flag:
            theta_j = np.zeros((3*len(ss),Nth))
            for i in range(Nth):
                print("data.X.shape = {}".format(data.X.shape))
                theta_j[:,i] = theta_list[i](np.concatenate((data.X[ss,data.last_idx[ss],:],data.X[ss,data.last_entry_idx[ss],:],data.X[ss,data.first_exit_idx[ss],:]),axis=0)).flatten()
            theta_yj = theta_j[:len(ss)]
            theta_xpj = theta_j[len(ss):2*len(ss)]
            theta_ypj = theta_j[2*len(ss):3*len(ss)]
            ans = (theta_x,theta_yj,theta_xpj,theta_ypj)
        return ans
    def plot_projections_1d_array(self,model,data,theta1d_list):
        q = model.q
        funlib = model.observable_function_library()
        Nth = len(theta1d_list)
        #n = q['Nz']-1
        comm_fwd = self.dam_moments['one']['xb'][0]
        eps = 1e-5
        mfpt_xb = self.dam_moments['one']['xb'][1,:,0]*(comm_fwd[:,0] > eps)/(comm_fwd[:,0] + 1*(comm_fwd[:,0] <= eps))
        mfpt_xb[np.where(comm_fwd[:,0] <= eps)[0]] = np.nan
        nnanidx = np.where(comm_fwd[:,0] > eps)[0]
        # Plot the 1D projections
        ss = np.random.choice(np.arange(self.nshort),size=min(self.nshort,500000),replace=False)
        chomss = self.chom[ss]/np.sum(self.chom[ss])
        # Make an array of axes
        fig,axes = plt.subplots(ncols=Nth,nrows=2,figsize=(6*Nth,6*2),sharex='col',sharey='row')
        for i in range(len(theta1d_list)):
            if theta1d_list[i] == 'LASSO':
                funname = 'LASSO'
                theta_x = theta_x = (self.lasso_predict_allz(model,data.X[ss,0,:])).reshape((len(ss),1))
                units = 1.0
                unit_symbol = ""
                file_suffix = "_LASSO"
            else:
                funname = funlib[theta1d_list[i]]["name"] #+ ' ({} km)'.format(altitude_list[j])
                theta_x = funlib[theta1d_list[i]]["fun"](data.X[ss,0,:])
                #theta_x = fun_at_level(funlib[theta1d_list[i]]["fun"],altitude_list[j],q)(data.X[ss,0,:])
                units = funlib[theta1d_list[i]]["units"]
                unit_symbol = funlib[theta1d_list[i]]["unit_symbol"]
                file_suffix = ("_{}".format(theta1d_list[i])).replace(".","p")
            xlim = np.array([np.min(theta_x),np.max(theta_x)])
            # Committor
            handles = []
            _,_,handle = self.plot_field_1d(comm_fwd[ss,0],chomss,theta_x,fieldname=r"$q^+$ (probability)",funname=funname,std_flag=True,label=r"$q^+$",color='black',units=units,unit_symbol=unit_symbol,fig=fig,ax=axes[0,i]) 
            axes[0,i].xaxis.set_visible(False)
            #fig.savefig(join(self.savefolder,"comm_fwd{}".format(file_suffix)))
            #plt.close(fig)
            # Conditional MFPT
            _,_,handle = self.plot_field_1d(mfpt_xb[ss],chomss,theta_x,fieldname=r"$\eta^+$ (days)",funname=funname,std_flag=True,label=r"$\eta^+$",color='black',units=units,unit_symbol=unit_symbol,fig=fig,ax=axes[1,i]) 
            if theta1d_list[i] == 'vT': 
                axes[1,i].xaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
            axes[1,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            axes[1,i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            if i > 0:
                axes[0,i].yaxis.set_visible(False)
                axes[1,i].yaxis.set_visible(False)
        axes[0,i].set_ylim([-0.1,1.1])
        axes[0,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        axes[0,i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        # Make sure there's a margin
        fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9)
        print("I subplots_adjusted. What more do you want?")
        fig.savefig(join(self.savefolder,"proj_1d_array"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        return
    #def compute_dam_moments_abba_finlag(self,model,data,function):
    #    Nx,Nt,xdim = data.X.shape
    #    dam_keys = list(model.dam_dict.keys())
    #    num_bvp = len(dam_keys)
    #    num_moments = self.num_moments
    #    bdy_dist = lambda x: np.minimum(model.adist(x),model.bdist(x))
    #    bdy_fun_b = lambda x: 1.0*(model.bdist(x) == 0)
    #    bdy_fun_a = lambda x: 1.0*(model.adist(x) == 0)
    #    bdy_fun_list_fwd = len(dam_keys)*[bdy_fun_b] + len(dam_keys)*[bdy_fun_a]
    #    bdy_fun_list_bwd = len(dam_keys)*[bdy_fun_a] + len(dam_keys)*[bdy_fun_b]
    #    dam_fun_list = 2*[model.dam_dict[k]['pay'] for k in dam_keys] 
    #    # Forward: x->B and x->A
    #    Fp,Pay,resp = function.solve_damage_function_moments_multiple(data,bdy_dist,bdy_fun_list_fwd,dam_fun_list,dirn=1,weights=np.ones(self.nshort)/self.nshort,num_moments=num_moments)
    #    # Backward: A->x and B->x
    #    Fm,_,resm = function.solve_damage_function_moments_multiple(data,bdy_dist,bdy_fun_list_bwd,dam_fun_list,dirn=-1,weights=self.chom,num_moments=num_moments)
    #    # Now combine them to get generalized rates
    #    Fmp = np.zeros((2*num_bvp,num_moments+1,Nx,Nt))
    #    Fmp[:,0] = Fm[:,0]*Fp[:,0]
    #    if num_moments >= 1:
    #        Fmp[:,1] = (1*Fm[:,1]*Fp[:,0] + 
    #                    1*Fm[:,0]*Fp[:,1])
    #    if num_moments >= 2:
    #        Fmp[:,2] = (1*Fm[:,2]*Fp[:,0] + 
    #                    2*Fm[:,1]*Fp[:,1] + 
    #                    1*Fm[:,0]*Fp[:,2])
    #    if num_moments >= 3:
    #        Fmp[:,3] = (1*Fm[:,3]*Fp[:,0] + 
    #                    3*Fm[:,2]*Fp[:,1] + 
    #                    3*Fm[:,1]*Fp[:,2] + 
    #                    1*Fm[:,0]*Fp[:,3])
    #    if num_moments >= 4:
    #        Fmp[:,4] = (1*Fm[:,4]*Fp[:,0] + 
    #                    4*Fm[:,3]*Fp[:,1] + 
    #                    6*Fm[:,2]*Fp[:,2] + 
    #                    4*Fm[:,1]*Fp[:,3] + 
    #                    1*Fm[:,0]*Fp[:,4])
    #    # Unweighted averages
    #    adist_x = (model.adist(data.X.reshape((Nt*Nx,xdim)))).reshape((Nx,Nt))
    #    bdist_x = (model.bdist(data.X.reshape((Nt*Nx,xdim)))).reshape((Nx,Nt))
    #    bdy_dist_x = np.minimum(adist_x,bdist_x)
    #    ramp_ab = Fp[0,0,:,:] #adist_x / (adist_x + bdist_x)
    #    ramp_ba = 1-Fp[0,0,:,:] #Fm[0,0,:,:] #bdist_x / (adist_x + bdist_x)
    #    #data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_seq[-1])
    #    #data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current)
    #    MFp = np.zeros((2*num_bvp,num_moments+1,Nx))
    #    #fd_total_decay = 0.5
    #    #fd_weights = fd_total_decay**((np.arange(len(self.lag_time_seq))-1)/(len(self.lag_time_seq)-1))
    #    #print("fd_weights = {}".format(fd_weights))
    #    #fd_decay_rate = fd_total_decay**(1/(len(self.lag_time_seq)-1))
    #    # TODO: figure this the heck out
    #    for i in range(num_moments+1):
    #        for j in range(len(self.lag_time_seq)-1):
    #            data.insert_boundaries_fwd(bdy_dist_x,data.t_x[j],data.t_x[-1])
    #            data.insert_boundaries_bwd(bdy_dist_x,data.t_x[j],data.t_x[0])
    #            MFp[:num_bvp,i,:] += Fp[:num_bvp,i,np.arange(data.nshort),data.first_exit_idx_fwd]*Fm[:num_bvp,i,np.arange(data.nshort),data.first_exit_idx_bwd]*(ramp_ab[:,j+1] - ramp_ab[:,j])
    #            MFp[num_bvp:,i,:] += Fp[num_bvp:,i,np.arange(data.nshort),data.first_exit_idx_fwd]*Fm[num_bvp:,i,np.arange(data.nshort),data.first_exit_idx_bwd]*(ramp_ba[:,j+1] - ramp_ba[:,j])
    #            #MFp[:num_bvp,i,:] += (ramp_ab[:,j]*Fp[:num_bvp,i,:,j] - ramp_ab[:,0]*Fp[:num_bvp,i,:,0])/self.lag_time_seq[j] * (j<=data.first_exit_idx) * fd_weights[j]
    #            #MFp[num_bvp:,i,:] += (ramp_ba[:,j]*Fp[num_bvp:,i,:,j] - ramp_ba[:,0]*Fp[num_bvp:,i,:,0])/self.lag_time_seq[j] * (j<=data.first_exit_idx) * fd_weights[j]
    #            #normalizer += fd_weights[j]*(j<=data.first_exit_idx)
    #        #print("normalizer: shp={}, min={}, max={}, mean={}, std={}".format(normalizer.shape,np.min(normalizer),np.max(normalizer),np.mean(normalizer),np.std(normalizer)))
    #        MFp[:,i,:] *= 1.0/self.lag_time_seq[-1]
    #        #if i > 0:
    #        #    MFp[:num_bvp,i] += ramp_ab[:,0]*i*Pay[:num_bvp,:,0]*Fp[:num_bvp,i-1,:,0]
    #        #    MFp[num_bvp:,i] += ramp_ba[:,0]*i*Pay[num_bvp:,:,0]*Fp[num_bvp:,i-1,:,0]
    #    Fmp_unweighted = np.zeros((2*num_bvp,num_moments+1))
    #    Fmp_unweighted[:,0] = np.sum(self.chom*Fm[:,0,:,0]*MFp[:,0,:], 1)
    #    if num_moments >= 1:
    #        Fmp_unweighted[:,1] = np.sum(self.chom*(
    #            1*Fm[:,1,:,0]*MFp[:,0,:] + 
    #            1*Fm[:,0,:,0]*MFp[:,1,:]), 1)
    #    if num_moments >= 2:
    #        Fmp_unweighted[:,2] = np.sum(self.chom*(
    #            1*Fm[:,2,:,0]*MFp[:,0,:] + 
    #            2*Fm[:,1,:,0]*MFp[:,1,:] +
    #            1*Fm[:,0,:,0]*MFp[:,2,:]), 1)
    #    if num_moments >= 3:
    #        Fmp_unweighted[:,3] = np.sum(self.chom*(
    #            1*Fm[:,3,:,0]*MFp[:,0,:] + 
    #            3*Fm[:,2,:,0]*MFp[:,1,:] +
    #            3*Fm[:,1,:,0]*MFp[:,2,:] +
    #            1*Fm[:,0,:,0]*MFp[:,3,:]), 1)
    #    if num_moments >= 4:
    #        Fmp_unweighted[:,4] = np.sum(self.chom*(
    #            1*Fm[:,4,:,0]*MFp[:,0,:] + 
    #            4*Fm[:,3,:,0]*MFp[:,1,:] +
    #            6*Fm[:,2,:,0]*MFp[:,2,:] +
    #            4*Fm[:,1,:,0]*MFp[:,3,:] +
    #            1*Fm[:,0,:,0]*MFp[:,4,:]), 1)
    #    # Now separate into the AB and BA components
    #    self.dam_moments = {}
    #    for k in range(num_bvp):
    #        self.dam_moments[dam_keys[k]] = {}
    #        self.dam_moments[dam_keys[k]]['ax'] = Fm[k]
    #        self.dam_moments[dam_keys[k]]['bx'] = Fm[num_bvp+k]
    #        self.dam_moments[dam_keys[k]]['xb'] = Fp[k]
    #        self.dam_moments[dam_keys[k]]['xa'] = Fp[num_bvp+k]
    #        self.dam_moments[dam_keys[k]]['ab'] = Fmp[k]
    #        self.dam_moments[dam_keys[k]]['ba'] = Fmp[num_bvp+k]
    #        self.dam_moments[dam_keys[k]]['res_ax'] = resm[k]
    #        self.dam_moments[dam_keys[k]]['res_bx'] = resm[num_bvp+k]
    #        self.dam_moments[dam_keys[k]]['res_xa'] = resp[k]
    #        self.dam_moments[dam_keys[k]]['res_xb'] = resp[num_bvp+k]
    #        self.dam_moments[dam_keys[k]]['rate_ab'] = Fmp_unweighted[k]
    #        self.dam_moments[dam_keys[k]]['rate_ba'] = Fmp_unweighted[num_bvp+k]
    #    return
    def compute_mfpt_unconditional(self,model,data,function):
        # Compute mean passage times and/or other functions, not caring it hits first
        Nx,Nt,xdim = data.X.shape
        bdy_dist = lambda x: model.bdist(x)
        bdy_fun = lambda x: np.zeros(len(x))
        src_fun = lambda x: np.ones(len(x))
        pot_fun = lambda x: np.zeros(len(x))
        function.fit_data(data.X[:,0],bdy_dist)
        self.mfpt_b = function.solve_boundary_value_problem(data,bdy_dist,bdy_fun,src_fun,pot_fun,dirn=1)
        bdy_dist = lambda x: model.adist(x)
        bdy_fun = lambda x: np.zeros(len(x))
        src_fun = lambda x: np.ones(len(x))
        pot_fun = lambda x: np.zeros(len(x))
        function.fit_data(data.X[:,0],bdy_dist)
        self.mfpt_a = function.solve_boundary_value_problem(data,bdy_dist,bdy_fun,src_fun,pot_fun,dirn=1)
        return
    def compute_dam_moments_abba(self,model,data,function):
        num_moments = self.num_moments
        Nx,Nt,xdim = data.X.shape
        dam_keys = list(model.dam_dict.keys())
        num_bvp = len(dam_keys)
        bdy_dist = lambda x: np.minimum(model.adist(x),model.bdist(x))
        bdy_fun_b = lambda x: 1.0*(model.bdist(x) == 0)
        bdy_fun_a = lambda x: 1.0*(model.adist(x) == 0)
        bdy_fun_list_fwd = len(dam_keys)*[bdy_fun_b] + len(dam_keys)*[bdy_fun_a]
        bdy_fun_list_bwd = len(dam_keys)*[bdy_fun_a] + len(dam_keys)*[bdy_fun_b]
        dam_fun_list = 2*[model.dam_dict[k]['pay'] for k in dam_keys] 
        # Forward: x->B and x->A
        Fp,Pay,resp = function.solve_damage_function_moments_multiple(data,bdy_dist,bdy_fun_list_fwd,dam_fun_list,dirn=1,weights=np.ones(self.nshort)/self.nshort,num_moments=num_moments)
        # Backward: A->x and B->x
        Fm,_,resm = function.solve_damage_function_moments_multiple(data,bdy_dist,bdy_fun_list_bwd,dam_fun_list,dirn=-1,weights=self.chom,num_moments=num_moments)
        # Now combine them to get generalized rates
        Fmp = np.zeros((2*num_bvp,num_moments+1,Nx,Nt))
        Fmp[:,0] = Fm[:,0]*Fp[:,0]
        if num_moments >= 1:
            Fmp[:,1] = (1*Fm[:,1]*Fp[:,0] + 
                        1*Fm[:,0]*Fp[:,1])
        if num_moments >= 2:
            Fmp[:,2] = (1*Fm[:,2]*Fp[:,0] + 
                        2*Fm[:,1]*Fp[:,1] + 
                        1*Fm[:,0]*Fp[:,2])
        if num_moments >= 3:
            Fmp[:,3] = (1*Fm[:,3]*Fp[:,0] + 
                        3*Fm[:,2]*Fp[:,1] + 
                        3*Fm[:,1]*Fp[:,2] + 
                        1*Fm[:,0]*Fp[:,3])
        if num_moments >= 4:
            Fmp[:,4] = (1*Fm[:,4]*Fp[:,0] + 
                        4*Fm[:,3]*Fp[:,1] + 
                        6*Fm[:,2]*Fp[:,2] + 
                        4*Fm[:,1]*Fp[:,3] + 
                        1*Fm[:,0]*Fp[:,4])
        # Unweighted averages
        adist_x = (model.adist(data.X.reshape((Nt*Nx,xdim)))).reshape((Nx,Nt))
        bdist_x = (model.bdist(data.X.reshape((Nt*Nx,xdim)))).reshape((Nx,Nt))
        ramp_ab = Fp[0,0,:,:] #adist_x / (adist_x + bdist_x)
        ramp_ba = 1-Fp[0,0,:,:] #Fm[0,0,:,:] #bdist_x / (adist_x + bdist_x)
        data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_seq[-1])
        #data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current)
        MFp = np.zeros((2*num_bvp,num_moments+1,Nx))
        fd_total_decay = 0.5
        fd_weights = fd_total_decay**((np.arange(len(self.lag_time_seq))-1)/(len(self.lag_time_seq)-1))
        print("fd_weights = {}".format(fd_weights))
        fd_decay_rate = fd_total_decay**(1/(len(self.lag_time_seq)-1))
        for i in range(num_moments+1):
            # To approximate the generator, write as a weighted sum of finite differences
            #fd_weights = np.exp(-self.lag_time_seq)
            normalizer = np.zeros(Nx)
            print("data.first_exit_idx: shape={}, min={}, max={}, mean={}, std={}".format(data.first_exit_idx.shape,np.min(data.first_exit_idx),np.max(data.first_exit_idx),np.mean(data.first_exit_idx),np.std(data.first_exit_idx)))
            for j in range(1,len(self.lag_time_seq)):
                MFp[:num_bvp,i,:] += (ramp_ab[:,j]*Fp[:num_bvp,i,:,j] - ramp_ab[:,0]*Fp[:num_bvp,i,:,0])/self.lag_time_seq[j] * (j<=data.first_exit_idx) * fd_weights[j]
                MFp[num_bvp:,i,:] += (ramp_ba[:,j]*Fp[num_bvp:,i,:,j] - ramp_ba[:,0]*Fp[num_bvp:,i,:,0])/self.lag_time_seq[j] * (j<=data.first_exit_idx) * fd_weights[j]
                normalizer += fd_weights[j]*(j<=data.first_exit_idx)
            print("normalizer: shp={}, min={}, max={}, mean={}, std={}".format(normalizer.shape,np.min(normalizer),np.max(normalizer),np.mean(normalizer),np.std(normalizer)))
            MFp[:,i,:] *= 1.0/normalizer
            if i > 0:
                MFp[:num_bvp,i] += ramp_ab[:,0]*i*Pay[:num_bvp,:,0]*Fp[:num_bvp,i-1,:,0]
                MFp[num_bvp:,i] += ramp_ba[:,0]*i*Pay[num_bvp:,:,0]*Fp[num_bvp:,i-1,:,0]
        Fmp_unweighted = np.zeros((2*num_bvp,num_moments+1))
        Fmp_unweighted[:,0] = np.sum(self.chom*Fm[:,0,:,0]*MFp[:,0,:], 1)
        if num_moments >= 1:
            Fmp_unweighted[:,1] = np.sum(self.chom*(
                1*Fm[:,1,:,0]*MFp[:,0,:] + 
                1*Fm[:,0,:,0]*MFp[:,1,:]), 1)
        if num_moments >= 2:
            Fmp_unweighted[:,2] = np.sum(self.chom*(
                1*Fm[:,2,:,0]*MFp[:,0,:] + 
                2*Fm[:,1,:,0]*MFp[:,1,:] +
                1*Fm[:,0,:,0]*MFp[:,2,:]), 1)
        if num_moments >= 3:
            Fmp_unweighted[:,3] = np.sum(self.chom*(
                1*Fm[:,3,:,0]*MFp[:,0,:] + 
                3*Fm[:,2,:,0]*MFp[:,1,:] +
                3*Fm[:,1,:,0]*MFp[:,2,:] +
                1*Fm[:,0,:,0]*MFp[:,3,:]), 1)
        if num_moments >= 4:
            Fmp_unweighted[:,4] = np.sum(self.chom*(
                1*Fm[:,4,:,0]*MFp[:,0,:] + 
                4*Fm[:,3,:,0]*MFp[:,1,:] +
                6*Fm[:,2,:,0]*MFp[:,2,:] +
                4*Fm[:,1,:,0]*MFp[:,3,:] +
                1*Fm[:,0,:,0]*MFp[:,4,:]), 1)
        # Now separate into the AB and BA components
        self.dam_moments = {}
        for k in range(num_bvp):
            self.dam_moments[dam_keys[k]] = {}
            self.dam_moments[dam_keys[k]]['ax'] = Fm[k]
            self.dam_moments[dam_keys[k]]['bx'] = Fm[num_bvp+k]
            self.dam_moments[dam_keys[k]]['xb'] = Fp[k]
            self.dam_moments[dam_keys[k]]['xa'] = Fp[num_bvp+k]
            self.dam_moments[dam_keys[k]]['ab'] = Fmp[k]
            self.dam_moments[dam_keys[k]]['ba'] = Fmp[num_bvp+k]
            self.dam_moments[dam_keys[k]]['res_ax'] = resm[k]
            self.dam_moments[dam_keys[k]]['res_bx'] = resm[num_bvp+k]
            self.dam_moments[dam_keys[k]]['res_xa'] = resp[k]
            self.dam_moments[dam_keys[k]]['res_xb'] = resp[num_bvp+k]
            self.dam_moments[dam_keys[k]]['rate_ab'] = Fmp_unweighted[k]
            self.dam_moments[dam_keys[k]]['rate_ba'] = Fmp_unweighted[num_bvp+k]
        return
    def plot_lifecycle_correlations_bar(self,model,keys=None):
        if keys is None: keys = list(model.corr_dict.keys())
        names = [r"$A\to A$",r"$A\to B$",r"$B\to B$",r"$B\to A$"]
        index = ['aa','ab','bb','ba']
        # Correlations
        maxcorr = 0
        fig,ax = plt.subplots(nrows=len(keys),figsize=(6,3*len(keys)),sharex=True)
        for k in range(len(keys)):
            print("key = {}. corr_dga range = ({},{}). corr_emp range = ({},{})".format(keys[k],np.min(self.lifecycle_corr_dga[keys[k]]),np.max(self.lifecycle_corr_dga[keys[k]]),np.min(self.lifecycle_corr_emp[keys[k]]),np.max(self.lifecycle_corr_emp[keys[k]])))
            df = pd.DataFrame(index=index,data=dict({
                "Phase": names,
                "DGA": [self.lifecycle_corr_dga[keys[k]][idx] for idx in index],
                "DNS": [self.lifecycle_corr_emp[keys[k]][idx] for idx in index],
                "DGA_unc": [2*self.lifecycle_corr_dga_unc[keys[k]][idx] for idx in index],
                "DNS_unc": [2*self.lifecycle_corr_emp_unc[keys[k]][idx] for idx in index],
                }))
            maxcorr = 1.1*np.nanmax(np.abs(df[["DGA","DNS"]].to_numpy() + df[["DGA_unc","DNS_unc"]].to_numpy()))
            print(df)
            df.plot(kind="bar",x="Phase",y=['DNS','DGA'],yerr=df[["DNS_unc","DGA_unc"]].to_numpy().T,ax=ax[k],color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
            ax[k].set_title(r"$\Gamma = $%s"%model.corr_dict[keys[k]]['name'])
            ax[k].set_ylabel(r"Corr($\Gamma,q^+q^-$)")
            ax[k].plot(names,np.zeros(len(names)),linestyle='--',color='black')
            #maxcorr = max(maxcorr,max(np.max(np.abs(self.lifecycle_corr_dga[keys[k]])),np.max(np.abs(self.lifecycle_corr_emp[keys[k]]))))
        for k in range(len(keys)):
            ax[k].set_ylim([-maxcorr,maxcorr])
        #fig.suptitle("Lifecycle correlations")
        fig.savefig(join(self.savefolder,"lifecycle_corr"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # Means -- plot the logs!
        fig,ax = plt.subplots(nrows=len(keys),figsize=(6,3*len(keys)),tight_layout=True,sharex=True)
        for k in range(len(keys)):
            df = pd.DataFrame(index=index,data=dict({
                "Phase": names,
                "DGA": [self.lifecycle_mean_dga[keys[k]][idx] for idx in index],
                "DNS": [self.lifecycle_mean_emp[keys[k]][idx] for idx in index],
                "DGA_unc": [2*self.lifecycle_mean_dga_unc[keys[k]][idx] for idx in index],
                "DNS_unc": [2*self.lifecycle_mean_emp_unc[keys[k]][idx] for idx in index],
                }))
            maxmean = 1.1*np.nanmax(np.abs(df[["DGA","DNS"]].to_numpy() + df[["DGA_unc","DNS_unc"]].to_numpy()))
            print(df)
            df.plot(kind="bar",x="Phase",y=['DNS','DGA'],yerr=df[["DNS_unc","DGA_unc"]].to_numpy().T,ax=ax[k],color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
            ax[k].plot(names,np.ones(len(names)),linestyle='--',color='black')
            ax[k].set_title(r"$\Gamma = $%s"%model.corr_dict[keys[k]]['name'])
            ax[k].set_ylabel(r"Phase overrepresentation")
            ax[k].set_yscale('log')
        fig.savefig(join(self.savefolder,"lifecycle_mean"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        return
    def plot_lifecycle_correlations(self,model,keys=None):
        # TODO: make this a bar plot
        # After they've been computed, plot
        if keys is None: keys = list(model.corr_dict.keys())
        names = [r"$A\to A$",r"$A\to B$",r"$B\to B$",r"$B\to A$"]
        # Correlations
        maxcorr = 0
        fig,ax = plt.subplots(nrows=len(keys),figsize=(6,3*len(keys)),tight_layout=True,sharex=True)
        for k in range(len(keys)):
            print("key = {}. corr_dga range = ({},{}). corr_emp range = ({},{})".format(keys[k],np.min(self.lifecycle_corr_dga[keys[k]]),np.max(self.lifecycle_corr_dga[keys[k]]),np.min(self.lifecycle_corr_emp[keys[k]]),np.max(self.lifecycle_corr_emp[keys[k]])))
            hemp, = ax[k].plot(names,self.lifecycle_corr_emp[keys[k]],marker='o',color='black',label=r"DNS")
            hdga, = ax[k].plot(names,self.lifecycle_corr_dga[keys[k]],marker='o',color='red',label=r"DGA")
            ax[k].set_title(r"$\Gamma = $%s"%model.corr_dict[keys[k]]['name'])
            ax[k].set_ylabel(r"Corr($\Gamma,q^+q^-$)")
            #ax[k].set_ylim([-1,1])
            ax[k].plot(names,np.zeros(len(names)),linestyle='--',color='black')
            maxcorr = max(maxcorr,max(np.max(np.abs(self.lifecycle_corr_dga[keys[k]])),np.max(np.abs(self.lifecycle_corr_emp[keys[k]]))))
            ax[k].legend(handles=[hdga,hemp],prop={'size':13})
        for k in range(len(keys)):
            ax[k].set_ylim([-maxcorr,maxcorr])
        #fig.suptitle("Lifecycle correlations")
        fig.savefig(join(self.savefolder,"lifecycle_corr"))
        plt.close(fig)
        # Means
        fig,ax = plt.subplots(nrows=len(keys),figsize=(6,3*len(keys)),tight_layout=True,sharex=True)
        for k in range(len(keys)):
            hemp, = ax[k].plot(names,self.lifecycle_mean_emp[keys[k]],marker='o',color='black',label='DNS')
            hdga, = ax[k].plot(names,self.lifecycle_mean_dga[keys[k]],marker='o',color='red',label='DGA')
            ax[k].set_title(r"$\Gamma = $%s"%model.corr_dict[keys[k]]['name'])
            ax[k].set_ylabel(r"$\langle\Gamma,q^+q^-\rangle_\pi$")
            ax[k].legend(handles=[hdga,hemp],prop={'size':13})
        #fig.suptitle("Lifecycle correlations")
        fig.savefig(join(self.savefolder,"lifecycle_mean"))
        plt.close(fig)
        return
    def write_compare_lifecycle_correlations(self,model,data):
        # Write the correlations to a file
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        del x_long
        dt = t_long[1] - t_long[0]
        f = open(join(self.savefolder,'lifecycle.txt'),'w')
        keys = list(model.corr_dict.keys())
        dk0 = list(model.dam_dict.keys())[0]
        # Store a table for plotting
        self.lifecycle_corr_dga = {} 
        self.lifecycle_corr_dga_unc = {} 
        self.lifecycle_corr_emp = {}  
        self.lifecycle_corr_emp_unc = {} # error bar in sample mean
        self.lifecycle_mean_dga = {} 
        self.lifecycle_mean_dga_unc = {} 
        self.lifecycle_mean_emp = {}  
        self.lifecycle_mean_emp_unc = {}
        phase_symbols = ['aa','ab','bb','ba']
        phase_text_headers = ['A->A','A->B','B->B','B->A']
        bwd_symbols = ['ax','ax','bx','bx']
        fwd_symbols = ['xa','xb','xb','xa']
        bwd_ints = [-1,-1,1,1]
        fwd_ints = [-1,1,1,-1]
        for k in range(len(keys)):
            self.lifecycle_corr_dga[keys[k]] = {} #np.zeros(4)
            self.lifecycle_corr_dga_unc[keys[k]] = {} #np.zeros(4)
            self.lifecycle_mean_dga[keys[k]] = {} #np.zeros(4)
            self.lifecycle_mean_dga_unc[keys[k]] = {} #np.zeros(4)
            self.lifecycle_mean_emp[keys[k]] = {} #np.zeros(4)
            self.lifecycle_mean_emp_unc[keys[k]] = {} #np.zeros(4)
            self.lifecycle_corr_emp[keys[k]] = {} #np.zeros(4)
            self.lifecycle_corr_emp_unc[keys[k]] = {} #np.zeros(4)
            print("data.X.shape = {}".format(data.X.shape))
            Pay = model.corr_dict[keys[k]]['pay'](data.X[:,0,:]).flatten()
            print("Pay.shape = {}".format(Pay.shape))
            t_long,x_long = model.load_long_traj(self.long_simfolder)
            Pay_long = model.corr_dict[keys[k]]['pay'](x_long).flatten()
            del x_long
            print("Pay_long.shape = {}".format(Pay_long.shape))
            f.write("Correlation function %s\n"%model.corr_dict[keys[k]]['name'])
            for i in range(len(phase_symbols)):
                f.write("\t%s: "%(phase_text_headers[i]))
                # DGA
                comm_bwd = self.dam_moments[dk0][bwd_symbols[i]][0,:,0]
                comm_fwd = self.dam_moments[dk0][fwd_symbols[i]][0,:,0]
                reactive_flag = 1*(self.long_from_label==bwd_ints[i])*(self.long_to_label==fwd_ints[i])
                Zab = np.sum(self.chom*comm_bwd*comm_fwd)
                ZPay = np.sum(self.chom*Pay)
                mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)/(ZPay*Zab)
                corr_dga = (mean_trans_dga*ZPay*Zab - Zab*ZPay)/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
                self.lifecycle_corr_dga[keys[k]][phase_symbols[i]] = corr_dga #
                self.lifecycle_mean_dga[keys[k]][phase_symbols[i]] = mean_trans_dga #
                self.lifecycle_corr_dga_unc[keys[k]][phase_symbols[i]] = np.nan
                self.lifecycle_mean_dga_unc[keys[k]][phase_symbols[i]] = np.nan
                f.write("DGA: Z = %3.3e, mean = %3.3e, corr = %3.3e, "%(Zab,mean_trans_dga,corr_dga))
                # Empirical
                # First a point estimate
                #mean_trans_emp = np.sum(reactive_flag*Pay_long)/np.sum(reactive_flag)
                Z_emp = np.mean(reactive_flag)
                mean_trans_emp = np.mean(reactive_flag*Pay_long)/(np.mean(reactive_flag)*np.mean(Pay_long))
                corr_emp = (np.mean(reactive_flag*Pay_long) - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
                # Then uncertainty bounds. Do this by blocking up data, say into tenths
                Nlong = len(t_long)
                num_blocks = 10
                block_size = int(Nlong/num_blocks)
                Nlong = num_blocks*block_size
                block_indices = np.arange(Nlong).reshape((num_blocks,block_size))
                mean_trans_emp_blocks = np.zeros(num_blocks)
                corr_emp_blocks = np.zeros(num_blocks)
                Z_emp_blocks = np.zeros(num_blocks)
                for j in range(num_blocks):
                    idx = block_indices[j]
                    Z_emp_blocks[j] = np.mean(reactive_flag[idx])
                    mean_trans_emp_blocks[j] = np.mean(reactive_flag[idx]*Pay_long[idx])/(np.mean(reactive_flag[idx])*np.mean(Pay_long[idx]))
                    corr_emp_blocks[j] = (np.mean(reactive_flag[idx]*Pay_long[idx]) - np.mean(reactive_flag[idx])*np.mean(Pay_long[idx]))/np.sqrt(np.mean(reactive_flag[idx]**2)*np.mean(Pay_long[idx]**2))
                f.write("EMP: Z = %3.3e +/- %3.3e, mean = %3.3e, corr = %3.3e\n"%(Z_emp,np.std(Z_emp_blocks),mean_trans_emp,corr_emp))
                self.lifecycle_corr_emp[keys[k]][phase_symbols[i]] = corr_emp #
                self.lifecycle_mean_emp[keys[k]][phase_symbols[i]] = mean_trans_emp #
                self.lifecycle_mean_emp_unc[keys[k]][phase_symbols[i]] = np.std(mean_trans_emp_blocks)
                self.lifecycle_corr_emp_unc[keys[k]][phase_symbols[i]] = np.std(corr_emp_blocks)
        return
        #    # ------------- B -> A -------------
        #    f.write("\tB->A: ")
        #    comm_bwd = self.dam_moments[dk0]['bx'][0,:,0]
        #    comm_fwd = self.dam_moments[dk0]['xa'][0,:,0]
        #    reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==-1)
        #    Zba = np.sum(self.chom*comm_bwd*comm_fwd)
        #    mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)/Zba
        #    corr_dga = (mean_trans_dga*Zba - Zba*np.sum(self.chom*Pay))/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
        #    f.write("DGA: Z = %3.3e, mean = %3.3e, corr = %3.3e, "%(Zba,mean_trans_dga,corr_dga))
        #    mean_trans_emp = np.sum(reactive_flag*Pay_long)/np.sum(reactive_flag)
        #    corr_emp = (np.mean(reactive_flag*Pay_long) - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
        #    f.write("EMP: Z = %3.3e, mean = %3.3e, corr = %3.3e\n"%(np.mean(reactive_flag),mean_trans_emp,corr_emp))
        #    self.lifecycle_corr_dga[keys[k]][3] = corr_dga #[k,3] = corr_dga
        #    self.lifecycle_corr_emp[keys[k]][3] = corr_emp #[k,3] = corr_emp
        #    self.lifecycle_mean_dga[keys[k]][3] = mean_trans_dga #
        #    self.lifecycle_mean_emp[keys[k]][3] = mean_trans_emp #
        #    # ------------- A -> A ----------------
        #    f.write("\tA->A: ")
        #    comm_bwd = self.dam_moments[dk0]['ax'][0,:,0]
        #    comm_fwd = self.dam_moments[dk0]['xa'][0,:,0]
        #    reactive_flag = 1*(self.long_from_label==-1)*(self.long_to_label==-1)
        #    Zaa = np.sum(self.chom*comm_bwd*comm_fwd)
        #    mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)/Zaa
        #    corr_dga = (mean_trans_dga*Zaa - Zaa*np.sum(self.chom*Pay))/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
        #    f.write("DGA: Z = %3.3e, mean = %3.3e, corr = %3.3e, "%(Zaa,mean_trans_dga,corr_dga))
        #    mean_trans_emp = np.sum(reactive_flag*Pay_long)/np.sum(reactive_flag)
        #    corr_emp = (np.mean(reactive_flag*Pay_long) - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
        #    f.write("EMP: Z = %3.3e, mean = %3.3e, corr = %3.3e\n"%(np.mean(reactive_flag),mean_trans_emp,corr_emp))
        #    self.lifecycle_corr_dga[keys[k]][0] = corr_dga
        #    self.lifecycle_corr_emp[keys[k]][0] = corr_emp
        #    self.lifecycle_mean_dga[keys[k]][0] = mean_trans_dga #
        #    self.lifecycle_mean_emp[keys[k]][0] = mean_trans_emp #
        #    # ------------- B -> B ----------------
        #    f.write("\tB->B: ")
        #    comm_bwd = self.dam_moments[dk0]['bx'][0,:,0]
        #    comm_fwd = self.dam_moments[dk0]['xb'][0,:,0]
        #    reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==1)
        #    Zbb = np.sum(self.chom*comm_bwd*comm_fwd)
        #    mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)/Zbb
        #    corr_dga = (mean_trans_dga*Zbb - Zbb*np.sum(self.chom*Pay))/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
        #    f.write("DGA: Z = %3.3e, mean = %3.3e, corr = %3.3e, "%(Zbb,mean_trans_dga,corr_dga))
        #    #f.write("DGA: mean = %3.3e, corr = %3.3e, "%(mean_trans_dga,corr_dga))
        #    mean_trans_emp = np.sum(reactive_flag*Pay_long)/np.sum(reactive_flag)
        #    corr_emp = (np.mean(reactive_flag*Pay_long) - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
        #    f.write("EMP: Z = %3.3e, mean = %3.3e, corr = %3.3e\n"%(np.mean(reactive_flag),mean_trans_emp,corr_emp))
        #    #f.write("EMP: mean = %3.3e, corr = %3.3e\n"%(mean_trans_emp,corr_emp))
        #    self.lifecycle_corr_dga[keys[k]][2] = corr_dga
        #    self.lifecycle_corr_emp[keys[k]][2] = corr_emp
        #    self.lifecycle_mean_dga[keys[k]][2] = mean_trans_dga #
        #    self.lifecycle_mean_emp[keys[k]][2] = mean_trans_emp #
        #return
    def write_compare_generalized_rates(self,model,data,suffix=''):
        # Write out the generalized rates of each moment, each observable, to a file
        # Write both per unit time, and per trajectory
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        del x_long
        f = open(join(self.savefolder,'gen_rates{}.txt'.format(suffix)),'w')
        keys = list(model.dam_dict.keys())
        num_moments = len(self.dam_moments[keys[0]]['rate_ab']) - 1
        for k in range(len(keys)):
            units_k = model.dam_dict[keys[k]]['units']
            unit_symbol_k = model.dam_dict[keys[k]]['unit_symbol_t'] # After being multiplied by time
            units_t = model.dam_dict['one']['units']
            unit_symbol_t = model.dam_dict['one']['unit_symbol_t']
            # Collect the whole array for plotting
            dga_moments_trajwise = np.zeros((2,num_moments+1))
            emp_moments_trajwise = np.zeros((2,num_moments+1))
            emp_moments_trajwise_unc_lower = np.zeros((2,num_moments+1)) # for error bars
            emp_moments_trajwise_unc_upper = np.zeros((2,num_moments+1)) # for error bars
            self.dam_moments[keys[k]]['rate_avg'] = 0.5*(self.dam_moments[keys[k]]['rate_ab'] + self.dam_moments[keys[k]]['rate_ba'])
            f.write("Damage function %s\n"%model.dam_dict[keys[k]]['name_full'])
            f.write("\tA->B\n")
            dga_rate = self.dam_moments[keys[k]]['rate_avg'][0] #['rate_ab'][0]
            # For empirical rate, compute both the average and the uncertainty
            #emp_rate = len(self.dam_emp[keys[k]]['ab'])/(t_long[-1] - t_long[0])
            # Compute the correlation with T -- if the moments go up to 2
            if self.num_moments >= 2:
                # DGA
                egt_t_dga = np.sum(self.dam_moments[keys[k]]['ab'][1,:,0]*self.chom)/np.sum(self.dam_moments[keys[k]]['ab'][0,:,0]*self.chom)
                eg_dga = self.dam_moments[keys[k]]['rate_ab'][1]/dga_rate
                et_dga = self.dam_moments['one']['rate_ab'][1]/dga_rate
                vg_dga = self.dam_moments[keys[k]]['rate_ab'][2]/dga_rate - eg_dga**2
                vt_dga = self.dam_moments['one']['rate_ab'][2]/dga_rate - et_dga**2
                dga_corr = (egt_t_dga - eg_dga)*et_dga/np.sqrt(vg_dga*vt_dga)
                # Empirical
                egt_emp = np.mean(self.dam_emp[keys[k]]['ab'].flatten()*self.dam_emp['one']['ab'].flatten())
                eg_emp = np.mean(self.dam_emp[keys[k]]['ab'])
                et_emp = np.mean(self.dam_emp['one']['ab'])
                vg_emp = np.var(self.dam_emp[keys[k]]['ab'])
                vt_emp = np.var(self.dam_emp['one']['ab'])
                emp_corr = (egt_emp - eg_emp*et_emp)/np.sqrt(vg_emp*vt_emp)
                # Plot them
                fig,ax = plt.subplots()
                scat = ax.scatter(units_k*units_t*self.dam_emp[keys[k]]['ab'].flatten(),units_t*self.dam_emp['one']['ab'].flatten(),color='black',marker='.')
                # Plot two crosses, one empirical and one DGA
                hemp, = ax.plot(units_k*eg_emp*np.ones(2),units_t*(et_emp + np.sqrt(vt_emp)*np.array([-1,1])), color='cyan', linestyle='-',linewidth=4,label='DNS')
                ax.plot(units_k*(eg_emp + np.sqrt(vg_emp)*np.array([-1,1])), units_t*et_emp*np.ones(2), color='cyan', linestyle='-',linewidth=4,label='DNS')
                hdga, = ax.plot(units_k*eg_dga*np.ones(2),units_t*(et_dga + np.sqrt(vt_dga)*np.array([-1,1])), color='red', linestyle='-',linewidth=4,label='DGA')
                ax.plot(units_k*(eg_dga + np.sqrt(vg_dga)*np.array([-1,1])), units_t*et_dga*np.ones(2), color='red', linestyle='-',linewidth=4,label='DGA')
                ax.set_xlabel(r"$%s\,(%s)$"%(model.dam_dict[keys[k]]['name_full'],unit_symbol_k),fontdict=font)
                ax.set_ylabel(r"$%s\,(%s)$"%(model.dam_dict['one']['name_full'],unit_symbol_t),fontdict=font)
                ax.set_xlim([min(0,ax.get_xlim()[0]),ax.get_xlim()[1]])
                ax.set_ylim([min(0,ax.get_ylim()[0]),ax.get_ylim()[1]])
                xlim,ylim = ax.get_xlim(),ax.get_ylim()
                fmt_x = helper.generate_sci_fmt(xlim[0],xlim[1],numdiv=100)
                fmt_y = helper.generate_sci_fmt(ylim[0],ylim[1],numdiv=100)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                ax.set_title(r"Empirical integrals $A\to B$",fontdict=font)
                ax.legend(handles=[hemp,hdga])
                fig.savefig(join(self.savefolder,"corr{}_ab".format(keys[k])),bbox_inches="tight",pad_inches=0.2)
                plt.close(fig)
                f.write("\t\tCorrelation with T: DGA: %3.3e, EMP: %3.3e\n"%(dga_corr,emp_corr))
            f.write("\t\tRate: DGA: %3.3e, EMP: %3.3e +/- %3.3e; return time = %3.3e +/- %3.3e\n"%(dga_rate,self.emp_rate,self.emp_rate_unc,1/self.emp_rate,self.emp_return_time_unc))
            for i in range(num_moments+1):
                #print("self.dam_emp[keys[k]]['ab'].shape = {}".format(self.dam_emp[keys[k]]['ab'].shape))
                dga_avg_per_time_tweighted = units_k**i/units_t*np.sum(
                        self.dam_moments[keys[k]]['ab'][i,:,0]*self.chom)
                dga_avg_per_traj_tweighted = units_t*dga_avg_per_time_tweighted / np.sum(
                        self.dam_moments[keys[k]]['ab'][0,:,0]*self.chom)
                emp_avg_per_time_tweighted = units_k**i/units_t*np.sum(self.dam_emp[keys[k]]['ab'].flatten()**i*self.dam_emp['one']['ab'])/(t_long[-1] - t_long[0])
                emp_avg_per_traj_tweighted = units_t*emp_avg_per_time_tweighted*(t_long[-1] - t_long[0]) / np.sum(self.dam_emp['one']['ab'])
                dga_avg_per_time = units_k**i/units_t*self.dam_moments[keys[k]]['rate_ab'][i]
                dga_avg_per_traj = units_t*dga_avg_per_time / dga_rate
                emp_avg_per_time = units_k**i/units_t*np.sum(self.dam_emp[keys[k]]['ab']**i)/(t_long[-1] - t_long[0])
                emp_avg_per_traj = units_k**i*np.mean(self.dam_emp[keys[k]]['ab']**i)
                #f.write("\t\tMoment %d: DGA/time = %3.3e, DGA/traj = %3.3e, EMP/time = %3.3e, EMP/traj = %3.3e\n" % (i,dga_avg_per_time,dga_avg_per_traj,emp_avg_per_time,emp_avg_per_traj))
                f.write("\t\tMoment %d: DGA/time (t-weighted) = %3.3e, DGA/traj (t-weighted) = %3.3e,  DGA/time = %3.3e, DGA/traj = %3.3e, EMP/time (t-weighted) = %3.3e, EMP/traj (t-weighted) = %3.3e, EMP/time = %3.3e, EMP/traj = %3.3e\n" % (i,dga_avg_per_time_tweighted,dga_avg_per_traj_tweighted,dga_avg_per_time,dga_avg_per_traj,emp_avg_per_time_tweighted,emp_avg_per_traj_tweighted,emp_avg_per_time,emp_avg_per_traj))
                dga_moments_trajwise[0,i] = dga_avg_per_traj
                emp_moments_trajwise[0,i] = emp_avg_per_traj
                unc = units_k**i*helper.mean_uncertainty(self.dam_emp[keys[k]]['ab']**i)
                emp_moments_trajwise_unc_upper[0,i] = emp_avg_per_traj+unc
                emp_moments_trajwise_unc_lower[0,i] = emp_avg_per_traj-unc
            f.write("\tB->A\n")
            dga_rate = self.dam_moments[keys[k]]['rate_avg'][0] #['rate_ba'][0]
            #emp_rate = len(self.dam_emp[keys[k]]['ba'])/(t_long[-1] - t_long[0])
            # Compute the correlation with T
            if num_moments >= 2:
                # DGA
                egt_t_dga = np.sum(self.dam_moments[keys[k]]['ba'][1,:,0]*self.chom)/np.sum(self.dam_moments[keys[k]]['ba'][0,:,0]*self.chom)
                eg_dga = self.dam_moments[keys[k]]['rate_ba'][1]/dga_rate
                et_dga = self.dam_moments['one']['rate_ba'][1]/dga_rate
                vg_dga = self.dam_moments[keys[k]]['rate_ba'][2]/dga_rate - eg_dga**2
                vt_dga = self.dam_moments['one']['rate_ba'][2]/dga_rate - et_dga**2
                dga_corr = (egt_t_dga - eg_dga)*et_dga/np.sqrt(vg_dga*vt_dga)
                # Empirical
                egt_emp = np.mean(self.dam_emp[keys[k]]['ba'].flatten()*self.dam_emp['one']['ba'].flatten())
                eg_emp = np.mean(self.dam_emp[keys[k]]['ba'])
                et_emp = np.mean(self.dam_emp['one']['ba'])
                vg_emp = np.var(self.dam_emp[keys[k]]['ba'])
                vt_emp = np.var(self.dam_emp['one']['ba'])
                emp_corr = (egt_emp - eg_emp*et_emp)/np.sqrt(vg_emp*vt_emp)
                # Plot them
                fig,ax = plt.subplots()
                scat = ax.scatter(units_k*self.dam_emp[keys[k]]['ba'].flatten(),units_t*self.dam_emp['one']['ba'].flatten(),color='black',marker='.')
                # Plot two crosses, one empirical and one DGA
                hemp, = ax.plot(units_k*eg_emp*np.ones(2),units_t*(et_emp + np.sqrt(vt_emp)*np.array([-1,1])), color='cyan', linestyle='-',linewidth=4,label='DNS')
                ax.plot(units_k*(eg_emp + np.sqrt(vg_emp)*np.array([-1,1])), units_t*et_emp*np.ones(2), color='cyan', linestyle='-',linewidth=4,label='DNS')
                hdga, = ax.plot(units_k*eg_dga*np.ones(2),units_t*(et_dga + np.sqrt(vt_dga)*np.array([-1,1])), color='red', linestyle='-',linewidth=4,label='DGA')
                ax.plot(units_k*(eg_dga + np.sqrt(vg_dga)*np.array([-1,1])), units_t*et_dga*np.ones(2), color='red', linestyle='-',linewidth=4,label='DGA')
                ax.set_xlabel(r"$%s (%s)$"%(model.dam_dict[keys[k]]['name_full'],unit_symbol_k),fontdict=font)
                ax.set_ylabel(r"$%s (%s)$"%(model.dam_dict['one']['name_full'],unit_symbol_t),fontdict=font)
                xlim,ylim = ax.get_xlim(),ax.get_ylim()
                fmt_x = helper.generate_sci_fmt(xlim[0],xlim[1])
                fmt_y = helper.generate_sci_fmt(ylim[0],ylim[1])
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                ax.set_title(r"Empirical integrals $B\to A$",fontdict=font)
                ax.legend(handles=[hemp,hdga])
                fig.savefig(join(self.savefolder,"corr{}_ba".format(keys[k])),bbox_inches="tight",pad_inches=0.2)
                plt.close(fig)
                f.write("\t\tCorrelation with T: DGA: %3.3e, EMP: %3.3e\n"%(dga_corr,emp_corr))
            f.write("\t\tRate: DGA: %3.3e, EMP: %3.3e +/- %3.3e; return time = %3.3e +/- %3.3e\n"%(dga_rate,self.emp_rate,self.emp_rate_unc,1/self.emp_rate,self.emp_return_time_unc))
            for i in range(num_moments+1):
                dga_avg_per_time_tweighted = units_k**i/units_t*np.sum(
                        self.dam_moments[keys[k]]['ba'][i,:,0]*self.chom)
                dga_avg_per_traj_tweighted = units_t*dga_avg_per_time_tweighted / np.sum(
                        self.dam_moments[keys[k]]['ba'][0,:,0]*self.chom)
                emp_avg_per_time_tweighted = units_k**i/units_t*np.sum(self.dam_emp[keys[k]]['ba'].flatten()**i*self.dam_emp['one']['ba'])/(t_long[-1] - t_long[0])
                emp_avg_per_traj_tweighted = units_t*emp_avg_per_time_tweighted*(t_long[-1] - t_long[0]) / np.sum(self.dam_emp['one']['ba'])
                dga_avg_per_time = units_k**i/units_t*self.dam_moments[keys[k]]['rate_ba'][i]
                dga_avg_per_traj = units_t*dga_avg_per_time / dga_rate
                emp_avg_per_time = units_k**i/units_t*np.sum(self.dam_emp[keys[k]]['ba']**i)/(t_long[-1] - t_long[0])
                emp_avg_per_traj = units_k**i*np.mean(self.dam_emp[keys[k]]['ba']**i)
                # Also estimate the error in emp_avg_per_traj
                f.write("\t\tMoment %d: DGA/time (t-weighted) = %3.3e, DGA/traj (t-weighted) = %3.3e,  DGA/time = %3.3e, DGA/traj = %3.3e, EMP/time (t-weighted) = %3.3e, EMP/traj (t-weighted) = %3.3e, EMP/time = %3.3e, EMP/traj = %3.3e\n" % (i,dga_avg_per_time_tweighted,dga_avg_per_traj_tweighted,dga_avg_per_time,dga_avg_per_traj,emp_avg_per_time_tweighted,emp_avg_per_traj_tweighted,emp_avg_per_time,emp_avg_per_traj))
                dga_moments_trajwise[1,i] = dga_avg_per_traj
                emp_moments_trajwise[1,i] = emp_avg_per_traj
                unc = units_k**i*helper.mean_uncertainty(self.dam_emp[keys[k]]['ba']**i)
                emp_moments_trajwise_unc_upper[1,i] = emp_avg_per_traj+unc
                emp_moments_trajwise_unc_lower[1,i] = emp_avg_per_traj-unc
            # Plot the moments for validation -- this time with error bars!
            fig,ax = plt.subplots(ncols=2,figsize=(12,6),sharey=False)
            bounds = np.array([emp_moments_trajwise_unc_lower[0,1:]**(1/np.arange(1,num_moments+1)),emp_moments_trajwise_unc_upper[0,1:]**(1/np.arange(1,num_moments+1))])
            y = emp_moments_trajwise[0,1:]**(1/np.arange(1,num_moments+1))
            yerr = np.array([y-bounds[0],bounds[1]-y])
            print("y = {}, yerr = {}".format(y,yerr))
            # TODO: make a data frame and plot the moments as bars
            # ---- dataframe plotting method ------
            df_moments = pd.DataFrame(index=np.arange(1,num_moments+1),
                    data = {
                        "Moment": np.arange(1,num_moments+1),
                        "DGA": dga_moments_trajwise[0,1:]**(1/np.arange(1,num_moments+1)),
                        "DGA_errlo": np.nan*np.ones(num_moments),
                        "DGA_errhi": np.nan*np.ones(num_moments),
                        "DNS": y,
                        "DNS_errlo": 2*yerr[0],
                        "DNS_errhi": 2*yerr[1],
                        })
            print(df_moments)
            df_moments.plot(x="Moment",y=["DNS","DGA"],yerr=df_moments[["DNS_errlo","DNS_errhi","DGA_errlo","DGA_errhi"]].to_numpy().T.reshape((2,2,num_moments)),kind='bar',ax=ax[0],color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
            # ---- Line plotting method -----------
            #hemp, = ax[0].plot(np.arange(1,num_moments+1),y,color='black',marker='o',markersize=20,linewidth=2,label='DNS')
            #hdga, = ax[0].plot(np.arange(1,num_moments+1),dga_moments_trajwise[0,1:]**(1/np.arange(1,num_moments+1)),color='red',marker='o',markersize=20,linewidth=2,label='DGA')
            #ax[0].errorbar(np.arange(1,num_moments+1),y,yerr=yerr,ecolor='black',elinewidth=2,capsize=3.0,color='black',marker='o',markersize=20,linewidth=2,zorder=10)
            #ax[0].legend(handles=[hdga,hemp],prop={'size':18})
            # ----------------------------
            ax[0].set_title(r"%s $(A\to B)$ moments"%model.dam_dict[keys[k]]['name'],fontdict=font)
            ax[0].set_xlabel("Moment number $k$",fontdict=font)
            ax[0].set_ylabel(r"$\left\{E\left[\left(%s\right)^k\right]\right\}^{1/k}$"%model.dam_dict[keys[k]]['name_full'],fontdict=font)
            ax[0].tick_params(axis='both',labelsize=14)
            # -------- B -> A moments -----------
            bounds = np.array([emp_moments_trajwise_unc_lower[1,1:]**(1/np.arange(1,num_moments+1)),emp_moments_trajwise_unc_upper[1,1:]**(1/np.arange(1,num_moments+1))])
            y = emp_moments_trajwise[1,1:]**(1/np.arange(1,num_moments+1))
            yerr = np.array([y-bounds[0],bounds[1]-y])
            print("y = {}, yerr = {}".format(y,yerr))
            # ---- dataframe plotting method ------
            df_moments = pd.DataFrame(index=np.arange(1,num_moments+1),
                    data = {
                        "Moment": np.arange(1,num_moments+1),
                        "DGA": dga_moments_trajwise[1,1:]**(1/np.arange(1,num_moments+1)),
                        "DGA_errlo": np.nan*np.ones(num_moments),
                        "DGA_errhi": np.nan*np.ones(num_moments),
                        "DNS": y,
                        "DNS_errlo": 2*yerr[0],
                        "DNS_errhi": 2*yerr[1],
                        })
            print(df_moments)
            df_moments.plot(x="Moment",y=["DNS","DGA"],yerr=df_moments[["DNS_errlo","DNS_errhi","DGA_errlo","DGA_errhi"]].to_numpy().T.reshape((2,2,num_moments)),kind='bar',ax=ax[1],color=['cyan','red'],rot=0,error_kw=dict(ecolor='black',lw=3,capsize=6,capthick=3))
            # ----- Line plotting method ----------
            #hemp, = ax[1].plot(np.arange(1,num_moments+1),y,color='black',marker='o',markersize=20,linewidth=2,label='DNS')
            #hdga, = ax[1].plot(np.arange(1,num_moments+1),dga_moments_trajwise[1,1:]**(1/np.arange(1,num_moments+1)),color='red',marker='o',markersize=20,linewidth=2,label='DNS')
            #ax[1].errorbar(np.arange(1,num_moments+1),y,yerr=yerr,ecolor='black',elinewidth=2,capsize=3.0,color='black',marker='o',markersize=20,linewidth=2,zorder=10)
            #ax[1].legend(handles=[hdga,hemp],prop={'size':18})
            # ---------------------------
            ax[1].set_title(r"%s $(B\to A)$ moments"%model.dam_dict[keys[k]]['name'],fontdict=font)
            ax[1].set_xlabel(r"Moment number $k$",fontdict=font)
            #ax[1].set_ylabel(r"$E[(\int_B^A%s\,dt)^n]$"%model.dam_dict[keys[k]]['pay_symbol'],fontdict=font)
            ax[1].tick_params(axis='both',labelsize=14)
            fig.savefig(join(self.savefolder,"moments_abba_log_{}".format(model.dam_dict[keys[k]]['abb_full'])),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # ----------------- Inferring PDFs -----------------
            # Also plot the full distribution and its approximation as Gamma
            fig,ax = plt.subplots(ncols=2,figsize=(12,6),tight_layout=True,sharey=True)
            # A -> B 
            hist,bin_edges = np.histogram(self.dam_emp[keys[k]]['ab'],bins=15,density=True)
            emp_mean = np.mean(self.dam_emp[keys[k]]['ab'])
            emp_meansq = np.mean(self.dam_emp[keys[k]]['ab']**2)
            alpha_emp,beta_emp = helper.gamma_mom(np.array([emp_mean,emp_meansq]))
            bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
            # Gamma distribution
            loc = self.dam_moments[keys[k]]['rate_ab'][1]/dga_rate
            esq = self.dam_moments[keys[k]]['rate_ab'][2]/dga_rate
            var = esq - loc**2
            #alpha = loc**2/var
            #beta = loc/var
            alpha,beta = helper.gamma_mom(self.dam_moments[keys[k]]['rate_ab'][1:]/dga_rate)
            
            print("AB, EMP: alpha={}, beta={}".format(alpha_emp,beta_emp))
            print("AB, DGA: alpha={}, beta={}".format(alpha,beta))
            dga_gamma_ab = beta**alpha/special.gamma(alpha)*bin_centers**(alpha-1)*np.exp(-beta*bin_centers)
            emp_gamma_ab = beta_emp**alpha_emp/special.gamma(alpha_emp)*bin_centers**(alpha_emp-1)*np.exp(-beta_emp*bin_centers)
            hemp_pdf_ab, = ax[0].plot(bin_centers,hist,color='black',marker='o',label='DNS PDF')
            #hemp_gamma_ab, = ax[0].plot(bin_centers,emp_gamma_ab,color='blue',marker='o',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha_emp),helper.sci_fmt_latex(beta_emp)))
            hdga_gamma_ab, = ax[0].plot(bin_centers,dga_gamma_ab,color='red',marker='o',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha),helper.sci_fmt_latex(beta)))
            #ax[0].set_yscale('log')
            ax[0].set_title(r"$A\to B$ %s PDF"%model.dam_dict[keys[k]]["name"],fontdict=font)
            ax[0].legend(handles=[hemp_pdf_ab,hdga_gamma_ab],prop={'size': 16})
            #ax[0].legend(handles=[hemp_pdf_ab,hdga_gamma_ab],prop={'size': 16})
            ax[0].tick_params(axis='both',labelsize=14)
            ax[0].set_xlabel(r"%s ($%s$)"%(model.dam_dict[keys[k]]["name"],model.dam_dict[keys[k]]["unit_symbol"]),fontdict=font)
            #  B -> A
            hist,bin_edges = np.histogram(self.dam_emp[keys[k]]['ba'],bins=15,density=True)
            emp_mean = np.mean(self.dam_emp[keys[k]]['ba'])
            emp_meansq = np.mean(self.dam_emp[keys[k]]['ba']**2)
            alpha_emp,beta_emp = helper.gamma_mom(np.array([emp_mean,emp_meansq]))
            bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
            loc = self.dam_moments[keys[k]]['rate_ba'][1]/dga_rate
            esq = self.dam_moments[keys[k]]['rate_ba'][2]/dga_rate
            var = esq - loc**2
            #alpha = loc**2/var
            #beta = loc/var
            alpha,beta = helper.gamma_mom(self.dam_moments[keys[k]]['rate_ba'][1:]/dga_rate)
            print("BA, EMP: alpha={}, beta={}".format(alpha_emp,beta_emp))
            print("BA, DGA: alpha={}, beta={}".format(alpha,beta))
            dga_gamma_ba = beta**alpha/special.gamma(alpha)*bin_centers**(alpha-1)*np.exp(-beta*bin_centers)
            emp_gamma_ba = beta_emp**alpha_emp/special.gamma(alpha_emp)*bin_centers**(alpha_emp-1)*np.exp(-beta_emp*bin_centers)
            hemp_pdf_ba, = ax[1].plot(bin_centers,hist,color='black',marker='o',label='DNS PDF')
            #hemp_gamma_ba, = ax[1].plot(bin_centers,emp_gamma_ba,marker='o',color='blue',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha_emp),helper.sci_fmt_latex(beta_emp)))
            hdga_gamma_ba, = ax[1].plot(bin_centers,dga_gamma_ba,marker='o',color='red',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha),helper.sci_fmt_latex(beta)))
            #ax[1].set_yscale('log')
            ax[1].set_title(r"$B\to A$ %s PDF"%model.dam_dict[keys[k]]["name"],fontdict=font)
            ax[1].legend(handles=[hemp_pdf_ba,hdga_gamma_ba],prop={'size': 16})
            #ax[1].legend(handles=[hemp_pdf_ba,hdga_gamma_ba],prop={'size': 16})
            ax[1].tick_params(axis='both',labelsize=14)
            ax[1].set_xlabel(r"%s ($%s$)"%(model.dam_dict[keys[k]]["name"],model.dam_dict[keys[k]]["unit_symbol"]),fontdict=font)
            fig.savefig(join(self.savefolder,"pdf_{}".format(model.dam_dict[keys[k]]['abb_full'])),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
        f.close()
        return
    def display_dam_moments_abba_current(self,model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs,horz_lines=0):
        # Plot the whole shebang, including current and eventually FW and observed paths
        # But do it in increments: least action path, then sample paths, then reactive density, then reactive current
        # Load the transitions
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        ab_reactive_flag = 1*(self.long_from_label==-1)*(self.long_to_label==1)
        ba_reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==-1)
        ab_starts = np.where(np.diff(ab_reactive_flag)==1)[0] + 1
        ab_ends = np.where(np.diff(ab_reactive_flag)==-1)[0] + 1
        ba_starts = np.where(np.diff(ba_reactive_flag)==1)[0] + 1
        ba_ends = np.where(np.diff(ba_reactive_flag)==-1)[0] + 1
        # Randomly select a few transitions
        num_obs = min([10,len(ab_starts),len(ba_starts)])
        ab_obs_idx = np.arange(num_obs) #np.random.choice(np.arange(len(ab_starts)),num_obs)
        ba_obs_idx = np.arange(num_obs) #np.random.choice(np.arange(len(ba_starts)),num_obs)
        theta_ab_obs = []
        theta_ba_obs = []
        for i in range(num_obs):
            theta_ab_obs += [theta_2d_fun(x_long[ab_starts[ab_obs_idx[i]]:ab_ends[ab_obs_idx[i]]])]
            theta_ba_obs += [theta_2d_fun(x_long[ba_starts[ba_obs_idx[i]]:ba_ends[ba_obs_idx[i]]])]
        del x_long
        # If I substitute in F- and F+ for q- and q+, I guess we'll see where pathways accumulate the most of whatever damage function it's measuring
        Nx,Nt,xdim = data.X.shape
        ss = np.random.choice(np.arange(Nx),10000,replace=False)
        keys = list(model.dam_dict.keys())
        num_moments = self.dam_moments[keys[0]]['xb'].shape[0]-1
        theta_xst = theta_2d_fun(model.tpt_obs_xst) # Possibly to be used as theta_ab
        eps = 0.001
        adist = model.adist(data.X.reshape((Nx*Nt,xdim)))
        bdist = model.bdist(data.X.reshape((Nx*Nt,xdim)))
        theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim)))
        interior_idx = np.where((adist>eps)*(bdist>eps))[0]
        thmin,thmax = np.min(theta_x[interior_idx,:],axis=0),np.max(theta_x[interior_idx,:],axis=0)
        theta_x = theta_x.reshape((Nx,Nt,2))
        for k in range(1): # num_moments):
            # -----------------------------
            # A->A
            print(f"------------ Starting A->A stuff -----------")
            comm_bwd = self.dam_moments[keys[k]]['ax'][0]
            comm_fwd = self.dam_moments[keys[k]]['xa'][0]
            weight = self.chom
            #theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
            xfw,tfw = model.load_least_action_path(self.physical_param_folder,dirn=1)
            theta_fw = theta_2d_fun(xfw)
            # Committor
            fieldname = r"$A\to A$"  #r"$\pi_{AB},J_{AB}$"
            field = comm_bwd*comm_fwd 
            field[(comm_fwd > eps)*(comm_fwd < 1-eps)*(comm_bwd > eps)*(comm_bwd < 1-eps) == 0] = np.nan
            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,current_bdy_flag=True,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.YlOrBr,theta_ab=None,abpoints_flag=False,ss=ss)
            if horz_lines > 0:
                print("---------------Drawing in some horizontal lines for flux distributions")
                # Draw in lines for reactive flux densities
                nnidx = np.where(np.isnan(field) == 0)[0]
                th1_min,th1_max = thmin[1],thmax[1]
                dramp = (th1_max - th1_min)/horz_lines
                print("th1_min = {}, th1_max = {}".format(th1_min,th1_max))
                th_levels = np.linspace(th1_min+0.5*dramp,th1_max-0.5*dramp,horz_lines)
                for i_th in range(len(th_levels)):
                    ax.axhline(y=th_levels[i_th]*theta_2d_units[1],color='black',linewidth=0.75,zorder=10)
            fig.savefig(join(self.savefolder,"jaa_rdens_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # -----------------------------
            # B->B
            print(f"------------ Starting B->B stuff -----------")
            comm_bwd = self.dam_moments[keys[k]]['bx'][0]
            comm_fwd = self.dam_moments[keys[k]]['xb'][0]
            weight = self.chom
            #theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
            xfw,tfw = model.load_least_action_path(self.physical_param_folder,dirn=1)
            theta_fw = theta_2d_fun(xfw)
            # Committor
            fieldname = r"$B\to B$"  #r"$\pi_{AB},J_{AB}$"
            field = comm_bwd*comm_fwd 
            field[(comm_fwd > eps)*(comm_fwd < 1-eps)*(comm_bwd > eps)*(comm_bwd < 1-eps) == 0] = np.nan
            #field *= (comm_fwd > 0)*(comm_fwd < 1)
            #field[(comm_fwd > 0)*(comm_fwd < 1) == 0] = np.nan
            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,current_bdy_flag=True,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.YlOrBr,theta_ab=None,abpoints_flag=False,ss=ss)
            if horz_lines > 0:
                print("---------------Drawing in some horizontal lines for flux distributions")
                # Draw in lines for reactive flux densities
                nnidx = np.where(np.isnan(field) == 0)[0]
                th1_min,th1_max = thmin[1],thmax[1]
                dramp = (th1_max - th1_min)/horz_lines
                print("th1_min = {}, th1_max = {}".format(th1_min,th1_max))
                th_levels = np.linspace(th1_min+0.5*dramp,th1_max-0.5*dramp,horz_lines)
                for i_th in range(len(th_levels)):
                    ax.axhline(y=th_levels[i_th]*theta_2d_units[1],color='black',linewidth=0.75,zorder=10)
            fig.savefig(join(self.savefolder,"jbb_rdens_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # -----------------------------
            # A->B
            print(f"------------ Starting A->B stuff -----------")
            comm_bwd = self.dam_moments[keys[k]]['ax'][0]
            comm_fwd = self.dam_moments[keys[k]]['xb'][0]
            weight = self.chom
            #theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
            xfw,tfw = model.load_least_action_path(self.physical_param_folder,dirn=1)
            theta_fw = theta_2d_fun(xfw)
            # Committor
            fieldname = r"$A\to B$"  #r"$\pi_{AB},J_{AB}$"
            field = comm_bwd * comm_fwd #self.dam_moments[keys[k]]['xb'][0] 
            field[(comm_fwd > eps)*(comm_fwd < 1-eps)*(comm_bwd > eps)*(comm_bwd < 1-eps) == 0] = np.nan
            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,current_bdy_flag=True,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=theta_fw,magu_obs=theta_ab_obs,cmap=plt.cm.YlOrBr,theta_ab=None,abpoints_flag=True,ss=ss)
            if horz_lines > 0:
                print("---------------Drawing in some horizontal lines for flux distributions")
                # Draw in lines for reactive flux densities
                nnidx = np.where(np.isnan(field) == 0)[0]
                th1_min,th1_max = thmin[1],thmax[1]
                dramp = (th1_max - th1_min)/horz_lines
                print("th1_min = {}, th1_max = {}".format(th1_min,th1_max))
                th_levels = np.linspace(th1_min+0.5*dramp,th1_max-0.5*dramp,horz_lines)
                for i_th in range(len(th_levels)):
                    ax.axhline(y=th_levels[i_th]*theta_2d_units[1],color='black',linewidth=0.75,zorder=10)
            fig.savefig(join(self.savefolder,"jab_rdens_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # ---------------------------------
            # B->A
            print(f"------------ Starting B->A stuff -----------")
            fieldname = r"$B\to A$ density, current" #r"$\pi_{BA},J_{BA}$"
            comm_bwd = self.dam_moments[keys[k]]['bx'][0]
            comm_fwd = self.dam_moments[keys[k]]['xa'][0]
            weight = self.chom
            #theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
            xfw,tfw = model.load_least_action_path(self.physical_param_folder,dirn=-1)
            theta_fw = theta_2d_fun(xfw)
            # Committor
            fieldname = r"$B\to A$"  #r"$\pi_{AB},J_{AB}$"
            field = comm_bwd * comm_fwd #self.dam_moments[keys[k]]['xa'][0] 
            field[(comm_fwd > eps)*(comm_fwd < 1-eps)*(comm_bwd > eps)*(comm_bwd < 1-eps) == 0] = np.nan
            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,current_bdy_flag=True,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=theta_fw,magu_obs=theta_ba_obs,cmap=plt.cm.YlOrBr,theta_ab=None,abpoints_flag=True)
            if horz_lines > 0:
                print("---------------Drawing in some horizontal lines for flux distributions")
                # Draw in lines for reactive flux densities
                nnidx = np.where(np.isnan(field) == 0)[0]
                th1_min,th1_max = thmin[1],thmax[1]
                dramp = (th1_max - th1_min)/horz_lines
                print("th1_min = {}, th1_max = {}".format(th1_min,th1_max))
                th_levels = np.linspace(th1_min+0.5*dramp,th1_max-0.5*dramp,horz_lines)
                for i_th in range(len(th_levels)):
                    ax.axhline(y=th_levels[i_th]*theta_2d_units[1],color='black',linewidth=0.75,zorder=10)
            fig.savefig(join(self.savefolder,"jba_rdens_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
        return
    def display_casts_abba(self,model,data,theta_2d_abbs):
        funlib = model.observable_function_library()
        Nx,Nt,xdim = data.X.shape
        keys = list(model.dam_dict.keys())
        num_moments = self.dam_moments[keys[0]]['xb'].shape[0]-1
        phase_flags = {'ab': True, 'ba': False, 'xa': False, 'xb': True, 'ax': True, 'bx': False}
        for i in range(len(theta_2d_abbs)):
            print("Starting view (%s,%s)"%(theta_2d_abbs[i][0],theta_2d_abbs[i][1]))
            fun0 = funlib[theta_2d_abbs[i][0]]
            fun1 = funlib[theta_2d_abbs[i][1]]
            theta_2d_fun = lambda x: np.array([fun0["fun"](x).flatten(),fun1["fun"](x).flatten()]).T
            #def theta_2d_fun(x):
            #    th = np.zeros((len(x),2))
            #    th[:,0] = fun0["fun"](x).flatten()
            #    th[:,1] = fun1["fun"](x).flatten()
            #    return th
            theta_xst = theta_2d_fun(model.tpt_obs_xst) # Possibly to be used as theta_ab
            theta_2d_names = [fun0["name"],fun1["name"]] #[r"$|\Psi(30 km)|$",r"$U(30 km)$"]
            theta_2d_units = np.array([fun0["units"],fun1["units"]])
            theta_2d_unit_symbols = [fun0["unit_symbol"],fun1["unit_symbol"]]
            weight = self.chom
            theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
            ## Plot unconditional MFPT
            #fig,ax = self.plot_field_2d(model,data,self.mfpt_b,weight,theta_x,shp=[20,20],fieldname=r"$E_x[\tau_B^+]$",fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
            #fsuff = 'mfpt_xb_th0%s_th1%s'%(theta_2d_abbs[i][0],theta_2d_abbs[i][1])
            #fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
            #plt.close(fig)
            #fig,ax = self.plot_field_2d(model,data,self.mfpt_a,weight,theta_x,shp=[20,20],fieldname=r"$E_x[\tau_A^+]$",fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
            #fsuff = 'mfpt_xa_th0%s_th1%s'%(theta_2d_abbs[i][0],theta_2d_abbs[i][1])
            #fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
            #plt.close(fig)


            for k in range(len(keys)):
                print("\tStarting damage function %s"%(keys[k]))
                field_units = model.dam_dict[keys[k]]['units']
                # Determine vmin and vmax
                if keys[k] == 'heatflux':
                    vmin,vmax = -10,200
                elif keys[k] == 'one':
                    vmin,vmax = 0,250
                else:
                    vmin,vmax = -np.inf,np.inf
                # Plot the actual function first
                field = field_units*model.dam_dict[keys[k]]['pay'](data.X[:,0]).reshape(-1,1)
                fieldname = model.dam_dict[keys[k]]['name']
                fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=None,comm_fwd=None,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                fsuff = '%s_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
                plt.close(fig)
                for j in range(min(3,self.num_moments)):
                    # A->B
                    if phase_flags['ab']:
                        comm_bwd = self.dam_moments[keys[k]]['ax'][0]
                        comm_fwd = self.dam_moments[keys[k]]['xb'][0]
                        if j == 0: 
                            fieldname = r"$q_A^-(x)q_B^+(x)$" #r"$P_x\{A\to B\}$"
                            field = field_units**j*self.dam_moments[keys[k]]['ab'][j] 
                        else:
                            prob = comm_bwd*comm_fwd
                            if j == 1: 
                                fieldname = r"$E_x[%s|A\to B]$"%(model.dam_dict[keys[k]]['name_full']) 
                                field = field_units**j*self.dam_moments[keys[k]]['ab'][j]
                            elif j == 2:
                                fieldname = r"$Var_x[%s|A\to B]$"%(model.dam_dict[keys[k]]['name_full']) 
                                field = field_units**j*(self.dam_moments[keys[k]]['ab'][j] - self.dam_moments[keys[k]]['ab'][1]**2)
                            field[np.where(prob==0)[0]] = np.nan
                            field *= 1.0/(prob + 1*(prob == 0))
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                            # Correct now
                            field[np.where(field > vmax**j)[0]] = np.nan
                            field[np.where(field < vmin**j)[0]] = np.nan
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                        fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                        fsuff = 'cast_%s%d_ab_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                        fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
                        plt.close(fig)
                        # Now plot divided by time
                        if j == 1 and keys[k] != 'one':
                            fieldname = r"$E_x[%s|A\to B]/E_x[%s|A\to B]$"%(model.dam_dict[keys[k]]['name_full'],model.dam_dict['one']['name_full'])
                            field = field/(self.dam_moments['one']['ab'][1])
                            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                            fsuff = 'castpertime_%s%d_ab_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                            fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0)
                            plt.close(fig)
                    # ---------------------------------
                    # B->A
                    if phase_flags['ba']:
                        comm_bwd = self.dam_moments[keys[k]]['bx'][0]
                        comm_fwd = self.dam_moments[keys[k]]['xa'][0]
                        if j == 0: 
                            fieldname = r"$P_x\{B\to A\}$"
                            field = field_units**j*self.dam_moments[keys[k]]['ba'][j] 
                        else:
                            prob = comm_bwd*comm_fwd
                            if j == 1: 
                                fieldname = r"$E_x[%s|B\to A]$"%(model.dam_dict[keys[k]]['name_full']) 
                                field = field_units**j*self.dam_moments[keys[k]]['ba'][j]
                            elif j == 2:
                                fieldname = r"$Var_x[%s|B\to A]$"%(model.dam_dict[keys[k]]['name_full']) 
                                field = field_units**j*(self.dam_moments[keys[k]]['ba'][j] - self.dam_moments[keys[k]]['ba'][1]**2)
                            field[np.where(prob==0)[0]] = np.nan
                            field *= 1.0/(prob + 1*(prob == 0))
                            # Bring within range only if j>=1
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                            # Correct now
                            field[np.where(field > vmax**j)[0]] = np.nan
                            field[np.where(field < vmin**j)[0]] = np.nan
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                        fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                        fsuff = 'cast_%s%d_ba_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                        fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
                        plt.close(fig)
                        # Now plot divided by time
                        if j == 1 and keys[k] != 'one':
                            fieldname = r"$E_x[%s|B\to A]/E_x[%s|B\to A]$"%(model.dam_dict[keys[k]]['name_full'],model.dam_dict['one']['name_full'])
                            field = field/(self.dam_moments['one']['ba'][1])
                            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                            fsuff = 'castpertime_%s%d_ba_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                            fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0)
                            plt.close(fig)
                    # x->B
                    if phase_flags['xb']:
                        comm_fwd = self.dam_moments[keys[k]]['xb'][0]
                        comm_bwd = self.dam_moments[keys[k]]['ax'][0]
                        if j == 0: 
                            fieldname = r"$q_B^+(x)$" #r"$P_x\{x\to B\}$"
                            field = field_units**j*self.dam_moments[keys[k]]['xb'][j] 
                        else:
                            prob = comm_fwd
                            if j == 1: 
                                if keys[k] == 'one':
                                    fieldname = r"$\eta^+_B(x)$"
                                else:
                                    fieldname = r"$E_x[%s|x\to B]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                                field = field_units**j*self.dam_moments[keys[k]]['xb'][j]
                            elif j == 2:
                                fieldname = r"$Var_x[%s|x\to B]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                                field = field_units**j*(self.dam_moments[keys[k]]['xb'][j] - self.dam_moments[keys[k]]['xb'][1]**2)
                            field[np.where(prob==0)[0]] = np.nan
                            field *= 1.0/(prob + 1*(prob == 0))
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                            # Correct now
                            field[np.where(field > vmax**j)[0]] = np.nan
                            field[np.where(field < vmin**j)[0]] = np.nan
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                        fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=True,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                        if keys[k] == 'one' and j == 1:
                            print("I got onto the one and j if statement")
                            _,_ = self.plot_field_2d(model,data,comm_fwd,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None,contourf_flag=False,contour_notf_flag=True,contour_notf_levels=np.array([0.1,0.2,0.5,0.8,0.9]),fig=fig,ax=ax)
                        fsuff = 'cast_%s%d_xb_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                        fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
                        plt.close(fig)
                    # ---------------------------------
                    # x->A
                    if phase_flags['xa']:
                        comm_fwd = self.dam_moments[keys[k]]['xa'][0]
                        comm_bwd = self.dam_moments[keys[k]]['bx'][0]
                        if j == 0: 
                            fieldname = r"$q_A^+(x)$" #r"$P_x\{x\to A\}$"
                            field = field_units**j*self.dam_moments[keys[k]]['xa'][j] 
                        else:
                            prob = comm_fwd
                            if j == 1: 
                                fieldname = r"$E_x[%s|x\to A]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                                field = field_units**j*self.dam_moments[keys[k]]['xa'][j]
                            elif j == 2:
                                fieldname = r"$Var_x[%s|x\to A]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                                field = field_units**j*(self.dam_moments[keys[k]]['xa'][j] - self.dam_moments[keys[k]]['xa'][1]**2)
                            field[np.where(prob==0)[0]] = np.nan
                            field *= 1.0/(prob + 1*(prob == 0))
                            # Bring within range only if j>=1
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                            # Correct now
                            field[np.where(field > vmax**j)[0]] = np.nan
                            field[np.where(field < vmin**j)[0]] = np.nan
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                        fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                        fsuff = 'cast_%s%d_xa_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                        fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
                        plt.close(fig)
                    # --------------------------
                    # B->x
                    if phase_flags['bx']:
                        comm_bwd = self.dam_moments[keys[k]]['bx'][0]
                        comm_fwd = self.dam_moments[keys[k]]['xa'][0]
                        if j == 0: 
                            fieldname = r"$q_B^-(x)$" #r"$P_x\{B\to x\}$"
                            field = field_units**j*self.dam_moments[keys[k]]['bx'][j] 
                        else:
                            prob = comm_bwd
                            if j == 1: 
                                if keys[k] == 'one':
                                    fieldname = r"$\eta_B^-$"
                                else:
                                    fieldname = r"$E_x[%s|B\to x]$"%(model.dam_dict[keys[k]]['name_bwd']) 
                                field = field_units**j*self.dam_moments[keys[k]]['bx'][j]
                            elif j == 2:
                                fieldname = r"$Var_x[%s|B\to x]$"%(model.dam_dict[keys[k]]['name_bwd']) 
                                field = field_units**j*(self.dam_moments[keys[k]]['bx'][j] - self.dam_moments[keys[k]]['bx'][1]**2)
                            field[np.where(prob==0)[0]] = np.nan
                            field *= 1.0/(prob + 1*(prob == 0))
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                            # Correct now
                            field[np.where(field > vmax**j)[0]] = np.nan
                            field[np.where(field < vmin**j)[0]] = np.nan
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                        fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                        fsuff = 'cast_%s%d_bx_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                        fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
                        plt.close(fig)
                    # ---------------------------------
                    # A->x
                    if phase_flags['ax']:
                        comm_bwd = self.dam_moments[keys[k]]['ax'][0]
                        comm_fwd = self.dam_moments[keys[k]]['xb'][0]
                        if j == 0: 
                            fieldname = r"$q_A^-(x)$" #r"$P_x\{A\to x\}$"
                            field = field_units**j*self.dam_moments[keys[k]]['ax'][j] 
                        else:
                            prob = comm_bwd
                            if j == 1: 
                                if keys[k] == 'one':
                                    fieldname = r"$-\eta_A^-$"
                                else:
                                    fieldname = r"$E_x[%s|A\to x]$"%(model.dam_dict[keys[k]]['name_bwd']) 
                                field = field_units**j*self.dam_moments[keys[k]]['ax'][j]
                            elif j == 2:
                                fieldname = r"$Var_x[%s|A\to x]$"%(model.dam_dict[keys[k]]['name_bwd']) 
                                field = field_units**j*(self.dam_moments[keys[k]]['ax'][j] - self.dam_moments[keys[k]]['ax'][1]**2)
                            field[np.where(prob==0)[0]] = np.nan
                            field *= 1.0/(prob + 1*(prob == 0))
                            # Bring within range only if j>=1
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                            # Correct now
                            field[np.where(field > vmax**j)[0]] = np.nan
                            field[np.where(field < vmin**j)[0]] = np.nan
                            print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                        fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                        fsuff = 'cast_%s%d_ax_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                        fig.savefig(join(self.savefolder,fsuff),bbox_inches="tight",pad_inches=0.2)
                        plt.close(fig)
        return
    def compute_naive_time(self,model,data,theta_fun,theta_name,theta_units,theta_unit_symbol):
        # For a given coordinate, compute the total expected running time to achieve 
        # Let it be 1D or 2D
        # theta_fun must be 1D
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        nlong = len(x_long)
        theta_short = theta_fun(data.X[:,0])
        theta_long = theta_fun(x_long)
        qb_long = 1*(self.long_to_label == 1)
        tDc_long = self.dam_emp['one']['x_Dc']
        qb_short = self.dam_moments['one']['xb'][0,:,0]
        tDc_short = self.dam_moments['one']['xb'][1,:,0] + self.dam_moments['one']['xa'][1,:,0]
        # Estimate variance according to both DGA and EMP, and see if it's comparable to the error between them
        # Get both things on a grid
        shp = [20,]
        shp,dth,thaxes,cgrid,tDc_long_grid,tDc_short_grid = helper.compare_fields(theta_long,theta_short,tDc_long,tDc_short,np.ones(nlong)/nlong,self.chom,shp=shp)
        _,_,_,_,qb_long_grid,qb_short_grid = helper.compare_fields(theta_long,theta_short,qb_long,qb_short,np.ones(nlong)/nlong,self.chom,shp=shp)
        #_,_,_,_,qb_var_long_grid,qb_var_short_grid = helper.compare_fields(theta_long,theta_short,qb_var_long,qb_var_short,np.ones(nlong)/nlong,self.chom,shp=shp)
        eps_grid = qb_short_grid - qb_long_grid # This is the ``error'' over the grid. Maybe should also design a sensitivity dependent on position
        qb_var_long_grid = qb_long_grid*(1 - qb_long_grid)
        qb_var_short_grid = qb_short_grid*(1 - qb_short_grid)
        print("qb_long_grid = {}".format(qb_long_grid))
        print("qb_short_grid = {}".format(qb_short_grid))
        print("eps_grid range = ({},{})".format(np.nanmin(eps_grid),np.nanmax(eps_grid)))
        print("sum(eps_grid == 0) = {}".format(np.sum(eps_grid==0)))
        Teps_short = qb_var_short_grid.flatten()*(eps_grid.flatten()!=0)/(eps_grid + 1*(eps_grid==0)).flatten()**2*tDc_short_grid.flatten()
        Teps_long = qb_var_long_grid.flatten()*(eps_grid.flatten()!=0)/(eps_grid + 1*(eps_grid==0)).flatten()**2*tDc_long_grid.flatten()
        # Now somehow plot Teps_short and Teps_long, and compare the absolute times
        thdim = len(thaxes)
        if thdim == 2:
            # TODO
            fig,ax = plt.subplots(nrows=3,figsize=(18,6),constrained_layout=True)
        if thdim == 1:
            fig,ax = plt.subplots(nrows=2,figsize=(6,12),constrained_layout=True,sharex=True)
            #ax[0].plot(thaxes[0]*theta_units,eps_grid.flatten(),color='black',marker='o')
            ax[0].set_ylabel(r"$\epsilon=q^+$ (DGA) $-\ q^+$ (empirical)",fontdict=font)
            ax[0].set_title("DGA error")
            hlong, = ax[1].plot(thaxes[0]*theta_units,Teps_long.flatten(),color='black',marker='o',label=r"$\frac{q^+(1-q^+)}{\epsilon^2}E[\tau^+]$ (EMP)")
            hshort, = ax[1].plot(thaxes[0]*theta_units,Teps_short.flatten(),color='red',marker='o',label=r"$\frac{q^+(1-q^+)}{\epsilon^2}E[\tau^+]$ (DGA)")
            ax[1].legend(handles=[hlong,hshort],prop={'size':15})
            ax[1].set_xlabel("%s (%s)"%(theta_name,theta_unit_symbol),fontdict=font)
            ax[1].set_title(r"Time until empirical error)$ < $DGA error")
            ax[1].set_ylabel("Simulation time",fontdict=font)
            fig.savefig(join(self.savefolder,"robustness"))
            plt.close(fig)
        print("Plotted robustness")
        return
    def demonstrate_committor_mfpt(self,model,data,theta2d_fun,theta2d_names,theta2d_units,theta2d_unit_symbols):
        qp = self.dam_moments['one']['xb'][0,:,0]
        tb = self.dam_moments['one']['xb'][1,:,0]*(qp > 1e-2)/(qp + 1*(qp <= 1e-2))
        tb[np.where(qp <= 1e-2)[0]] = np.nan
        ta = self.dam_moments['one']['xa'][1,:,0]*(1-qp > 1e-2)/(1-qp + 1*(1-qp <= 1e-2))
        ta[np.where(1-qp <= 1e-2)[0]] = np.nan
        theta2d = theta2d_fun(data.X[:,0])
        print("theta2D.shape = {}".format(theta2d.shape))
        # Project both committor and t_b onto both 1D spaces and the joint space
        shp = np.array([45,45])
        shp0,dth0,thaxes0,_,qth0,_,_,_,_ = self.project_field(qp,self.chom,theta2d[:,0:1],shp=shp[0:1])
        shp1,dth1,thaxes1,_,qth1,_,_,_,_ = self.project_field(qp,self.chom,theta2d[:,1:2],shp=shp[1:2])
        _,dth,thaxes,_,qth,_,_,_,_ = self.project_field(qp,self.chom,theta2d,shp=shp)
        _,_,_,_,tbth,_,_,_,_ = self.project_field(tb,self.chom,theta2d,shp=shp)
        # Identify two bins in CV space: (1) (q_th0,q_th1) = (1/2,1/3) and (2) (q_th0,q_th1) = (2/3,1/2)
        # First, are they both monotonic?
        dqth0 = np.diff(qth0)
        dqth1 = np.diff(qth1)
        if not (np.all(dqth0 > 0) or np.all(dqth0 < 0)): 
            print("WARNING: Bad subspace where q is not monotonic.\n\tthaxes0[0]={}\n\tqth0={}".format(thaxes0[0],qth0))
        if not (np.all(dqth1 > 0) or np.all(dqth1 < 0)): 
            print("WARNING: Bad subspace where q is not monotonic.\n\tthaxes1[0]={}\n\tqth1={}".format(thaxes1[0],qth1))
        # Now make sure they cross 1/2 at some point
        if (qth0[0]-0.5)*(qth0[-1]-0.5) > 0: 
            sys.exit("ERROR: Bad subspace where q is always on the same side of 0.5.\n\tthaxes0[0]={}\n\tqth0={}".format(thaxes0[0],qth0))
        if (qth1[0]-0.5)*(qth1[-1]-0.5) > 0: 
            sys.exit("ERROR: Bad subspace where q is always on the same side of 0.5.\n\tthaxes1[0]={}\n\tqth1={}".format(thaxes1[0],qth1))
        # Now find where they do cross 1/2
        th0_idx0 = np.where(np.abs(np.diff(np.sign(qth0-0.5)))==2)[0]
        if len(th0_idx0) > 1: print("WARNING: 0.5 is crossed more than once\n\tqth0={}".format(qth0))
        th0_idx0 = th0_idx0[0]
        th0_idx1 = th0_idx0 + 1
        th1_idx0 = np.where(np.abs(np.diff(np.sign(qth1-0.5)))==2)[0]
        if len(th1_idx0) > 1: print("WARNING: 0.5 is crossed more than once\n\tqth1={}".format(qth1))
        th1_idx0 = th1_idx0[0]
        th1_idx1 = th1_idx0 + 1
        # Now identify the two bins
        th0_halfbin = thaxes0[0][th0_idx0:th0_idx0+2]
        th1_halfbin = thaxes1[0][th1_idx0:th1_idx0+2]
        # Now select two boxes in CV space from which to pull from
        qth_qth0half = np.mean(qth.reshape(shp)[th0_idx0:th0_idx0+2,:],0)
        qth_qth1half = np.mean(qth.reshape(shp)[:,th1_idx0:th1_idx0+2],1)
        print("qth_qth0half = {}".format(qth_qth0half))
        print("qth_qth1half = {}".format(qth_qth1half))
        theta2d_test_idx_lower = np.zeros((2,2), dtype=int)
        theta2d_test_idx_lower[:,1] = th1_idx0
        nnidx = np.where(np.isnan(qth_qth1half)==0)[0]
        theta2d_test_idx_lower[0,0] = nnidx[np.argmin(np.abs(qth_qth1half[nnidx]-0.3))]
        theta2d_test_idx_lower[1,0] = nnidx[np.argmin(np.abs(qth_qth1half[nnidx]-0.7))]
        # Now draw two sets of samples from pi
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        ina_long = (model.adist(x_long)==0)
        inb_long = (model.bdist(x_long)==0)
        theta2d_long = theta2d_fun(x_long)
        del x_long
        long_test_idx_0 = np.where(
                (theta2d_long[:,0] >= thaxes0[0][theta2d_test_idx_lower[0,0]])*
                (theta2d_long[:,0] <= thaxes0[0][theta2d_test_idx_lower[0,0]+1])*
                (theta2d_long[:,1] >= thaxes1[0][theta2d_test_idx_lower[0,1]])*
                (theta2d_long[:,1] <= thaxes1[0][theta2d_test_idx_lower[0,1]+1]))[0]
        long_test_idx_1 = np.where(
                (theta2d_long[:,0] >= thaxes0[0][theta2d_test_idx_lower[1,0]])*
                (theta2d_long[:,0] <= thaxes0[0][theta2d_test_idx_lower[1,0]+1])*
                (theta2d_long[:,1] >= thaxes1[0][theta2d_test_idx_lower[1,1]])*
                (theta2d_long[:,1] <= thaxes1[0][theta2d_test_idx_lower[1,1]+1]))[0]
        print("long_test_idx_0.shape = {}".format(long_test_idx_0.shape))
        print("long_test_idx_1.shape = {}".format(long_test_idx_1.shape))
        print("mean theta over long_test_idx_0 = {}".format(np.mean(theta2d_long[long_test_idx_0]*theta2d_units,0)))
        print("mean theta over long_test_idx_1 = {}".format(np.mean(theta2d_long[long_test_idx_1]*theta2d_units,0)))
        print("dth = {}".format(theta2d_units*dth))
        test_theta_0 = np.array([np.mean(thaxes0[0][theta2d_test_idx_lower[0,0]:theta2d_test_idx_lower[0,0]+2]),np.mean(thaxes1[0][theta2d_test_idx_lower[0,1]:theta2d_test_idx_lower[0,1]+2])])
        test_theta_1 = np.array([np.mean(thaxes0[0][theta2d_test_idx_lower[1,0]:theta2d_test_idx_lower[1,0]+2]),np.mean(thaxes1[0][theta2d_test_idx_lower[1,1]:theta2d_test_idx_lower[1,1]+2])])
        print("test_theta_0={}, test_theta_1={}".format(theta2d_units*test_theta_0,theta2d_units*test_theta_1))
        test_qth = np.zeros(2)
        test_tbth = np.zeros(2)
        test_qth[0] = np.mean(qth.reshape(shp)[theta2d_test_idx_lower[0,0]:theta2d_test_idx_lower[0,0]+2,theta2d_test_idx_lower[0,1]:theta2d_test_idx_lower[0,1]+2])
        test_qth[1] = np.mean(qth.reshape(shp)[theta2d_test_idx_lower[1,0]:theta2d_test_idx_lower[1,0]+2,theta2d_test_idx_lower[1,1]:theta2d_test_idx_lower[1,1]+2])
        test_tbth[0] = np.mean(tbth.reshape(shp)[theta2d_test_idx_lower[0,0]:theta2d_test_idx_lower[0,0]+2,theta2d_test_idx_lower[0,1]:theta2d_test_idx_lower[0,1]+2])
        test_tbth[1] = np.mean(tbth.reshape(shp)[theta2d_test_idx_lower[1,0]:theta2d_test_idx_lower[1,0]+2,theta2d_test_idx_lower[1,1]:theta2d_test_idx_lower[1,1]+2])
        print("test_qth = {}".format(test_qth))
        print("test_theta_0 = {}".format(theta2d_units*test_theta_0))
        print("test_theta_1 = {}".format(theta2d_units*test_theta_1))

        # -------------------- Plots -----------------------

        # 1. Maps of committor and MFPT
        fig,ax = plt.subplots(ncols=3,figsize=(18,6),sharey=True) #,constrained_layout=True)
        helper.plot_field_2d(qp,self.chom,theta2d,shp=[40,40],fun0name=theta2d_names[0],fun1name=theta2d_names[1],units=theta2d_units,unit_symbols=theta2d_unit_symbols,fig=fig,ax=ax[1],std_flag=False,cbar_pad=0.03,fmt_x=fmt2,fmt_y=fmt)
        helper.plot_field_2d(tb,self.chom,theta2d,shp=[40,40],fun0name=theta2d_names[0],fun1name=theta2d_names[1],units=theta2d_units,unit_symbols=theta2d_unit_symbols,fig=fig,ax=ax[2],std_flag=False,cbar_pad=0.03,fmt_x=fmt2,fmt_y=fmt)
        helper.plot_field_2d(np.ones(self.nshort),self.chom,theta2d,shp=[40,40],fun0name=theta2d_names[0],fun1name=theta2d_names[1],units=theta2d_units,unit_symbols=theta2d_unit_symbols,fig=fig,ax=ax[0],avg_flag=False,std_flag=False,logscale=True,cbar_pad=0.03,fmt_x=fmt2,fmt_y=fmt,cmap=plt.cm.binary)
        ax[1].set_title(r"$q^+(x)$",fontdict=font)
        ax[1].tick_params(axis='both',labelsize=18)
        ax[1].xaxis.label.set_fontsize(18)
        ax[1].yaxis.label.set_fontsize(18)
        ax[1].yaxis.set_visible(False)
        ax[2].set_title(r"$\eta^+(x)$",fontdict=font)
        ax[2].tick_params(axis='both',labelsize=18)
        ax[2].xaxis.label.set_fontsize(18)
        ax[2].yaxis.label.set_fontsize(18)
        ax[2].yaxis.set_visible(False)
        ax[0].set_title(r"$\pi(x)$",fontdict=font)
        ax[0].tick_params(axis='both',labelsize=18)
        ax[0].xaxis.label.set_fontsize(18)
        ax[0].yaxis.label.set_fontsize(18)
        theta_ab = theta2d_fun(model.tpt_obs_xst)
        for i in range(3):
            #ax[i].plot((thaxes0[0][th0_idx0]+thaxes0[0][th0_idx0+1])/2*theta2d_units[0]*np.ones(2), thaxes1[0][[0,-1]]*theta2d_units[1],linestyle='--',color='black',zorder=10)
            ax[i].plot(thaxes0[0][[0,-1]]*theta2d_units[0], (thaxes1[0][th1_idx0]+thaxes1[0][th1_idx0+1])/2*theta2d_units[1]*np.ones(2),linestyle='--',color='black',zorder=10)
            ax[i].text(theta2d_units[0]*theta_ab[0,0],theta2d_units[1]*theta_ab[0,1],asymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=25,horizontalalignment='center',verticalalignment='center',zorder=100)
            ax[i].text(theta2d_units[0]*theta_ab[1,0],theta2d_units[1]*theta_ab[1,1],bsymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=25,horizontalalignment='center',verticalalignment='center',zorder=100)
            ax[i].text(theta2d_units[0]*test_theta_0[0],theta2d_units[1]*test_theta_0[1],r"$\theta_0$",bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=20,horizontalalignment='center',verticalalignment='center',zorder=100)
            ax[i].text(theta2d_units[0]*test_theta_1[0],theta2d_units[1]*test_theta_1[1],r"$\theta_1$",bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=20,horizontalalignment='center',verticalalignment='center',zorder=100)
        fig.savefig(join(self.savefolder,"committor_2d"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # 2. Plot the timeseries
        num_series = 100
        dt = t_long[1] - t_long[0]
        fig,ax = plt.subplots(nrows=2,figsize=(12,12),constrained_layout=True,sharex=True)
        ss0 = np.sort(long_test_idx_0[np.where(np.isnan(self.dam_emp['one']['x_Dc'][long_test_idx_0])==0)[0]])
        #ss0 = np.sort(ss0[np.random.choice(np.arange(len(ss0)),min(num_series,len(ss0)),replace=False)])
        ss0 = ss0[np.linspace(0,len(ss0)-1,min(num_series,len(ss0))).astype(int)]
        ss1 = np.sort(long_test_idx_1[np.where(np.isnan(self.dam_emp['one']['x_Dc'][long_test_idx_1])==0)[0]])
        #ss1 = np.sort(ss1[np.random.choice(np.arange(len(ss1)),min(num_series,len(ss1)),replace=False)])
        ss1 = ss1[np.linspace(0,len(ss1)-1,min(num_series,len(ss0))).astype(int)]
        print("len(ss0) = {}; len(ss1) = {}".format(len(ss0),len(ss1)))
        Nb = np.zeros(2,dtype=int)
        Nb[0] = np.sum(self.long_to_label[ss0]==1)
        Nb[1] = np.sum(self.long_to_label[ss1]==1)
        tb_emp = np.zeros(2)
        tb_emp[0] = np.nanmean(self.dam_emp['one']['x_Dc'][ss0])
        tb_emp[1] = np.nanmean(self.dam_emp['one']['x_Dc'][ss1])
        max_length = 0
        for j in range(num_series):
            # First from theta0
            ti_min = ss0[j]
            tmax = self.dam_emp['one']['x_Dc'][ss0[j]]
            max_length = max(max_length,tmax)
            ti_max = ss0[j] + int(tmax/dt) 
            if self.long_to_label[ss0[j]] == 1:
                #label = r"$\frac{N_B}{N}=%.2f;\ P\{\theta_{%d}\to B\}=%.2f$"%(Nb[0]/num_series,0,test_qth[0])
                #label = r"$P\{\theta_{%d}\to B\}=%.2f$"%(0,test_qth[0])
                label = r"$q^+(\theta_{%d})=%.2f$"%(0,test_qth[0])
                h0b, = ax[0].plot(t_long[ti_min:ti_max+1]-t_long[ti_min],theta2d_long[ti_min:ti_max+1,1]*theta2d_units[1],color='red',label=label)
            else:
                h0a, = ax[0].plot(t_long[ti_min:ti_max+1]-t_long[ti_min],theta2d_long[ti_min:ti_max+1,1]*theta2d_units[1],color='deepskyblue',alpha=0.5,label=r"$\frac{N_A}{N}=%.2f$"%(1-Nb[0]/num_series))
            # Second from theta1
            ti_min = ss1[j]
            tmax = self.dam_emp['one']['x_Dc'][ss1[j]]
            max_length = max(max_length,tmax)
            ti_max = ss1[j] + int(tmax/dt) 
            if self.long_to_label[ss1[j]] == 1:
                #label = r"$\frac{N_B}{N}=%.2f;\ P\{\theta_{%d}\to B\}=%.2f$"%(Nb[1]/num_series,1,test_qth[1])
                #label = r"$P\{\theta_{%d}\to B\}=%.2f$"%(1,test_qth[1])
                label = r"$q^+(\theta_{%d})=%.2f$"%(1,test_qth[1])
                h1b, = ax[1].plot(t_long[ti_min:ti_max+1]-t_long[ti_min],theta2d_long[ti_min:ti_max+1,1]*theta2d_units[1],color='red',label=label)
            else:
                h1a, = ax[1].plot(t_long[ti_min:ti_max+1]-t_long[ti_min],theta2d_long[ti_min:ti_max+1,1]*theta2d_units[1],color='deepskyblue',label=r"$\frac{N_A}{N}=%.2f$"%(1-Nb[1]/num_series))
        handles = [[h0b],[h1b]]
        for j in range(2):
            if j == 1: ax[j].set_xlabel("Time (days)",fontdict=bigfont)
            ax[j].set_ylabel(theta2d_names[1],fontdict=bigfont)
            ax[j].set_title(r"Initial conditions $\theta_{%d}$"%(j),fontdict=bigfont)
            ax[j].plot([0,max_length],theta2d_units[1]*theta_ab[0,1]*np.ones(2),color='black',linestyle='-',linewidth=3)
            ax[j].plot([0,max_length],theta2d_units[1]*theta_ab[1,1]*np.ones(2),color='black',linestyle='-',linewidth=3)
            htj_dga, = ax[j].plot(test_tbth[j]*np.ones(2),theta2d_units[1]*theta_ab[:,1],color='black',linestyle='--',linewidth=4,label=r"$\eta^+(\theta_{%d})=%.1f$"%(j,test_tbth[j]))
            #htj_emp, = ax[j].plot(tb_emp[j]*np.ones(2),theta2d_units[1]*theta_ab[:,1],color='black',linestyle='--',linewidth=4,label=r"$\overline{\tau_B}(\theta_{%d})=%.1f$"%(j,tb_emp[j]))
            print("test_tbth[j]={}".format(test_tbth[j]))
            print("tb_emp[j]={}".format(tb_emp[j]))
            dthab = np.abs(theta_ab[0,1]-theta_ab[1,1])
            ax[j].text(0,theta2d_units[1]*(theta_ab[0,1]+0.01*dthab),asymb,fontdict=bigfont,color='black',weight='bold')
            ax[j].text(0,theta2d_units[1]*(theta_ab[1,1]+0.01*dthab),bsymb,fontdict=bigfont,color='black',weight='bold')
            ax[j].tick_params(axis='both',labelsize=25)
            ax[j].legend(handles=handles[j]+[htj_dga],prop={'size':25},loc="center right")
            print(r"$E[q^+|\theta_{%d}]=%.2f;\ \frac{N_B}{N}(\theta_{%d})=%.2f$"
            "\n"
            r"$E[\tau^+|\theta_{%d}\to B]=%.1f;\ \overline{T_B}(\theta_{%d})=%.1f$"
            %(j,test_qth[j],j,Nb[j]/num_series,j,test_tbth[j],j,tb_emp[j]))
            #ax[j].text(max_length*4/8,theta2d_units[1]*(theta_ab[0,1]*0.5+theta_ab[1,1]*0.5),r"$E[q^+|\theta_{%d}]=%.2f;\ \frac{N_B}{N}(\theta_{%d})=%.2f$"
            #"\n"
            #r"$E[\tau^+|\theta_{%d}\to B]=%.1f;\ \overline{T_B}(\theta_{%d})=%.1f$"
            #%(j,test_qth[j],j,Nb[j]/num_series,j,test_tbth[j],j,tb_emp[j]),fontdict=font)
        fig.savefig(join(self.savefolder,"committor_demo_timeseries"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        return
    def display_equiv_sim_time(self,model,data,theta_1d_name,theta_1d_units,theta_1d_unit_symbol,basis_size,theta_1d_abb,theta_1d_short=None,theta_1d_fun=None):
        # Just do the robustness comparison for a 1D observable function
        shp_1d = np.array([20,])
        keys = list(model.dam_dict.keys())
        for k in range(len(keys)):
            qp = self.dam_moments[keys[k]]['xb'][0,:,0]
            print("committor for %s: %s"%(keys[k],describe(qp)))
        num_moments = self.dam_moments[keys[0]]['xb'].shape[0]-1
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        if theta_1d_fun is None:
            long_subset = np.random.choice(np.arange(len(x_long)),size=10000,replace=False)
        else:
            long_subset = np.arange(len(x_long))
        t_long = t_long[long_subset]
        x_long = x_long[long_subset]
        nlong = len(x_long)
        tDc_long = self.dam_emp['one']['x_Dc'][long_subset]
        tDc_short = self.dam_moments['one']['xb'][1,:,0] + self.dam_moments['one']['xa'][1,:,0]
        if theta_1d_fun is not None:
            theta_1d_short = theta_1d_fun(data.X[:,0]).reshape(-1,1)
            theta_1d_long = theta_1d_fun(x_long).reshape(-1,1)
        else:
            # Need to interpolate
            theta_1d_long = self.out_of_sample_extension(theta_1d_short,data,x_long).reshape(-1,1)
            theta_1d_short = theta_1d_short.reshape(-1,1)
        for k in range(len(keys)):
            np.random.seed(0)
            # ------------------
            # x->B
            # Get both committors onto the same grid
            q_long = 1*(self.long_to_label[long_subset] == 1)
            q_short = self.dam_moments['one']['xb'][0,:,0]
            q_short = np.minimum(1,np.maximum(0,q_short))
            _,dth,thaxes,_,tDc_long_grid,tDc_short_grid = helper.compare_fields(theta_1d_long,theta_1d_short,tDc_long,tDc_short,np.ones(nlong)/nlong,self.chom,shp=shp_1d,subset_flag=False)
            _,_,_,_,q_long_grid,q_short_grid = helper.compare_fields(theta_1d_long,theta_1d_short,q_long,q_short,np.ones(nlong)/nlong,self.chom,shp=shp_1d,subset_flag=False)
            _,_,_,_,w_long_grid,w_short_grid = helper.compare_fields(theta_1d_long,theta_1d_short,np.ones(nlong),np.ones(self.nshort),np.ones(nlong)/nlong,self.chom,shp=shp_1d,avg_flag=False,subset_flag=False)
            _,_,_,_,N_long_grid,_,_,_,_ = helper.project_field(np.ones(nlong),np.ones(nlong),theta_1d_long,shp=shp_1d,avg_flag=False)
            print("N_long_grid = {}".format(N_long_grid))
            #sys.exit()
            eps_grid = np.abs(q_short_grid - q_long_grid)
            total_error = np.sqrt(np.nansum((q_long_grid-q_short_grid)**2*w_long_grid)/np.nansum(w_long_grid))
            q_var_long_grid = q_long_grid*(1 - q_long_grid)
            q_var_short_grid = q_short_grid*(1 - q_short_grid)
            #Teps_short = q_var_short_grid*(eps_grid!=0)/(eps_grid+1*(eps_grid==0))**2*tDc_short_grid
            #Teps_long = q_var_long_grid*(eps_grid!=0)/(eps_grid+1*(eps_grid==0))**2*tDc_short_grid
            Teps_short = q_var_short_grid/(total_error)**2*tDc_short_grid
            Teps_long = q_var_long_grid/(total_error)**2*tDc_short_grid
            Ttot_short = np.nansum(Teps_short*w_short_grid)
            Ttot_long = np.nansum(Teps_long*w_long_grid)
            uname = r"$P\{x\to B\}$"

            # --------------------------------------------
            # Zero: Plot one against the other. Simple
            fig,ax = plt.subplots(figsize=(6,6))
            h, = ax.plot(q_long_grid,q_short_grid,color='red',marker='o',label=r"RMS error \epsilon$=%.3f$"%(total_error))
            ax.plot([0,1],[0,1],color='black',linestyle='--')
            #ax.plot(q_long_grid+np.sqrt(q_var_long_grid/N_long_grid),q_long_grid,linestyle='--',color='black')
            #ax.plot(q_long_grid-np.sqrt(q_var_long_grid/N_long_grid),q_long_grid,linestyle='--',color='black')
            ax.legend(handles=[h],prop={'size': 14},loc='upper left')
            ax.set_xlabel(r"$q^+$ DNS", fontdict=font)
            ax.set_ylabel(r"$q^+$ DGA", fontdict=font)
            ax.set_title(r"$N=%s,\ M=%d$, Lag$=$%d days"%(helper.sci_fmt_latex0(self.nshort),basis_size,self.lag_time_seq[-1]),fontdict=font)
            fig.savefig(join(self.savefolder,"fidelity_qp"),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # ---------------------------------------------
            # One: Plot them both as a function of U (30 km)
            fig,ax = plt.subplots(figsize=(6,6))
            hemp, = ax.plot(theta_1d_units*thaxes[0],q_long_grid,color='black',marker='o',label=r"$q^+_{\mathrm{DNS}}$")
            hdga, = ax.plot(theta_1d_units*thaxes[0],q_short_grid,color='red',marker='o',label=r"$q^+_{\mathrm{DGA}}$")
            herr, = ax.plot([],[],color='white',label=r"RMS error $\epsilon=%.3f$"%(total_error))
            #ax.plot(q_long_grid+np.sqrt(q_var_long_grid/N_long_grid),q_long_grid,linestyle='--',color='black')
            #ax.plot(q_long_grid-np.sqrt(q_var_long_grid/N_long_grid),q_long_grid,linestyle='--',color='black')
            ax.legend(handles=[hemp,hdga,herr],prop={'size': 14},loc='lower left')
            ax.set_xlabel(r"%s (%s)"%(theta_1d_name,theta_1d_unit_symbol), fontdict=font)
            ax.set_ylabel(r"$q^+$ DGA, DNS", fontdict=font)
            ax.set_title(r"$N=%s,\ M=%d$, Lag$=$%d days"%(helper.sci_fmt_latex0(self.nshort),basis_size,self.lag_time_seq[-1]),fontdict=font)
            fig.savefig(join(self.savefolder,"fidelity_th"),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)

            # --------------------------------------------
            # Two: plot the time to epsilon
            fig,ax0 = plt.subplots(figsize=(6,6))
            ss = np.where(np.isnan(eps_grid)==0)[0]
            ss = ss[np.where(eps_grid[ss]>1e-10)[0]]
            print("Are any subsetted points nan? {}".format(np.sum(np.isnan(eps_grid[ss]))))
            #ss = np.where(eps_grid>1e-10)[0]
            c0 = 'black'
            h0, = ax0.plot(theta_1d_units*thaxes[0][ss],Teps_short[ss],color=c0,marker='o',label=r"Time req. for Err(EMP)$\leq\epsilon=%.3f$"%(total_error))
            #ax0.scatter(theta_1d_units*thaxes[0][ss],Teps_short[ss],color=c0,marker='o',s=75*w_short_grid[ss]/np.max(w_short_grid[ss]))
            h1, = ax0.plot(theta_1d_units*thaxes[0][ss],Ttot_long*np.ones(len(ss)),color=c0,linestyle='--',label=r"Avg. time = $%s$ days"%helper.sci_fmt_latex0(Ttot_long))
            leg = ax0.legend(handles=[h0,h1],prop={'size':14},loc='lower center')
            for text in leg.get_texts(): plt.setp(text,color=c0)
            ax0.set_xlabel(theta_1d_name,fontdict=font)
            ax0.set_ylabel("Simulation time (days)",fontdict=font,color=c0)
            ax0.set_ylim([5e1,8e4])
            ax0.set_yscale('log')
            ax0.set_title(r"$N=%s,\ M=%d$, Lag$=$%d days"%(helper.sci_fmt_latex0(self.nshort),basis_size,self.lag_time_seq[-1]),fontdict=font)
            ax0.tick_params(axis='y',colors=c0)
            #ax[1].set_title("Sim time req. for Err$_{EMP}\leq $Err$_{DGA}$",fontdict=font)
            fig.savefig(join(self.savefolder,"DGA_vs_EMP_Teps_{}".format(theta_1d_abb)),bbox="tight",pad_inches=0.2)
            plt.close(fig)
            # --------------------------------------------
            #fig,ax = plt.subplots(ncols=2,figsize=(12,6))
            #ax0,ax = ax
            fig,ax0 = plt.subplots(figsize=(6,6))
            c0 = 'black'
            ss = np.where(eps_grid>1e-10)[0]
            heps, = ax0.plot(thaxes[0][ss]*theta_1d_units,eps_grid[ss],color=c0,label=r"|$q^+_{DGA}-q^+_{EMP}$|")
            ax0.scatter(thaxes[0][ss]*theta_1d_units,eps_grid[ss],color=c0,marker='o',s=75*w_short_grid[ss]/np.max(w_short_grid[ss]))
            heps_avg, = ax0.plot(thaxes[0][ss]*theta_1d_units,total_error*np.ones(len(ss)),color=c0,linestyle='--',label=r"RMS = %.1e"%total_error)
            ax0.legend(handles=[heps,heps_avg],prop={'size':12},loc='lower right')
            ax0.set_xlabel(theta_1d_name,fontdict=font)
            ax0.set_ylabel(r"DGA error $|q^+_{DGA} - q^+_{EMP}|$",fontdict=font)
            #ax0.set_title(r"Err$_{DGA}$")
            ax0.set_title(r"$N=%s,\ M=%d$, Lag$=$%d days"%(helper.sci_fmt_latex0(self.nshort),basis_size,self.lag_time_seq[-1]))
            ax0.set_ylim([5e-3,5e-1])
            ax0.set_yscale('log')
            # Now plot the required simulation time
            ax1 = ax0.twinx()
            c1 = 'red'
            h0, = ax1.plot(theta_1d_units*thaxes[0][ss],Teps_short[ss],color=c1,marker='o',label=r"Time req. for Err(EMP)$\leq$Err(DGA)")
            ax1.scatter(theta_1d_units*thaxes[0][ss],Teps_short[ss],color=c1,marker='o',s=75*w_short_grid[ss]/np.max(w_short_grid[ss]))
            h1, = ax1.plot(theta_1d_units*thaxes[0][ss],Ttot_short*np.ones(len(ss)),color=c1,linestyle='--',label=r"Avg. time = %.1e"%Ttot_short)
            leg = ax1.legend(handles=[h0,h1],prop={'size':12},loc='upper right')
            for text in leg.get_texts(): plt.setp(text,color=c1)
            ax1.set_xlabel(theta_1d_name,fontdict=font)
            ax1.set_ylabel("Simulation time (days)",fontdict=font,color=c1)
            ax1.set_ylim([5e1,8e4])
            ax1.set_yscale('log')
            ax1.tick_params(axis='y',colors=c1)
            #ax[1].set_title("Sim time req. for Err$_{EMP}\leq $Err$_{DGA}$",fontdict=font)
            fig.savefig(join(self.savefolder,"DGA_vs_EMP_{}".format(theta_1d_abb)))
            plt.close(fig)
        return
    def display_dam_moments_abba_validation(self,model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units):
        # This function is for rigorous testing. Not for production in a paper. Well okay, maybe
        shp_1d = np.array([20])
        shp_2d = np.array([20,40])
        keys = list(model.dam_dict.keys())
        # First order of business: make sure the committor is the same for the different damage functions
        for k in range(len(keys)):
            qp = self.dam_moments[keys[k]]['xb'][0,:,0]
            print("committor for %s: %s"%(keys[k],describe(qp)))
        if len(keys) > 1: print("max diff between committors for %s and %s: %s"%(keys[0],keys[1],describe(self.dam_moments[keys[0]]['xb'][0,:,0] - self.dam_moments[keys[1]]['xb'][0,:,0])))
        num_moments = self.dam_moments[keys[0]]['xb'].shape[0]-1
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        nlong = len(x_long)
        tDc_long = self.dam_emp['one']['x_Dc']
        tDc_short = self.dam_moments['one']['xb'][1,:,0] + self.dam_moments['one']['xa'][1,:,0]
        theta_1d_short = theta_1d_fun(data.X[:,0])
        theta_1d_long = theta_1d_fun(x_long)
        theta_2d_short = theta_2d_fun(data.X[:,0])
        theta_2d_long = theta_2d_fun(x_long)
        print("theta_2d_short.shape = {}".format(theta_2d_short.shape))
        print("theta_2d_long.shape = {}".format(theta_2d_long.shape))
        for i in range(num_moments+1):
            for k in range(len(keys)):
                np.random.seed(0)
                # ------------------
                # x->B
                if i == 0:
                    uname = r"$P\{x\to B\}$"
                else:
                    uname = r"$E[(%s)^{%d}1_B(X(\tau^+))]$"%(model.dam_dict[keys[k]]['name_fwd'],i)
                fsuff = model.dam_dict[keys[k]]['abb_fwd']+'%d_xb'%i
                logscale = model.dam_dict[keys[k]]['logscale']
                u0 = self.dam_moments[keys[k]]['xb'][i,:,0]
                u1 = self.dam_emp[keys[k]]['x_Dc']**i*(self.long_to_label==1)
                fig,ax = helper.compare_plot_fields_2d(theta_2d_short,theta_2d_long,u0,u1,self.chom,np.ones(nlong)/nlong,theta_names=theta_2d_names,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_2d)
                fig.savefig(join(self.savefolder,fsuff+'_2d'))
                plt.close(fig)
                fig,ax = helper.compare_plot_fields_1d(theta_1d_short,theta_1d_long,u0,u1,self.chom,np.ones(nlong)/nlong,theta_name=theta_1d_name,theta_units=theta_1d_units,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_1d)
                fig.savefig(join(self.savefolder,fsuff+'_1d'))
                plt.close(fig)
                # Now look at DNS equivalent running time
                q_long = 1*(self.long_to_label == 1)
                q_short = self.dam_moments['one']['xb'][0,:,0]
                q_short = np.minimum(1,np.maximum(0,q_short))
                # 1D
                _,_,thaxes,_,tDc_long_grid,tDc_short_grid = helper.compare_fields(theta_1d_long,theta_1d_short,tDc_long,tDc_short,np.ones(nlong)/nlong,self.chom,shp=shp_1d)
                _,_,_,_,q_long_grid,q_short_grid = helper.compare_fields(theta_1d_long,theta_1d_short,q_long,q_short,np.ones(nlong)/nlong,self.chom,shp=shp_1d)
                _,_,_,_,w_long_grid,w_short_grid = helper.compare_fields(theta_1d_long,theta_1d_short,np.ones(nlong),np.ones(self.nshort),np.ones(nlong)/nlong,self.chom,shp=shp_1d,avg_flag=False)
                eps_grid = q_short_grid - q_long_grid
                q_var_long_grid = q_long_grid*(1 - q_long_grid)
                q_var_short_grid = q_short_grid*(1 - q_short_grid)
                Teps_short = q_var_short_grid*(eps_grid!=0)/(eps_grid+1*(eps_grid==0))**2*tDc_short_grid
                Teps_long = q_var_long_grid*(eps_grid!=0)/(eps_grid+1*(eps_grid==0))**2*tDc_short_grid
                Ttot_short = np.sum(Teps_short*w_short_grid)
                Ttot_long = np.sum(Teps_long*w_long_grid)
                print("------------- key = {} --------------------".format(keys[k]))
                print("1D: Ttot_short = %f, Ttot_long = %f"%(Ttot_short,Ttot_long))
                fig,ax = plt.subplots(figsize=(6,6))
                #h0, = ax.plot(theta_1d_units*thaxes[0],Teps_long,color='black',marker='o',label=r"Pointwise time (Emp.)")
                h1, = ax.plot(theta_1d_units*thaxes[0],Teps_short,color='black',marker='o',label=r"Pointwise time")
                #h2, = ax.plot(theta_1d_units*thaxes[0][[0,-1]],Ttot_long*np.ones(2),color='black',linestyle='--',label=r"Avg. time (Emp.)")
                h3, = ax.plot(theta_1d_units*thaxes[0][[0,-1]],Ttot_short*np.ones(2),color='red',linestyle='--',label=r"Avg. time")
                ax.legend(handles=[h1,h3],prop={'size':12})
                ax.set_xlabel("%s"%(theta_1d_name),fontdict=font)
                ax.set_ylabel("Simulation time (days)",fontdict=font)
                ax.set_yscale('log')
                ax.set_title("Sim. time req. for $Err_{emp}\leq Err_{DGA}$",fontdict=font)
                fig.savefig(join(self.savefolder,fsuff+'_1d_Teps_withline'))
                plt.close(fig)
                #sys.exit()
                # 2D
                _,_,thaxes,_,tDc_long_grid,tDc_short_grid = helper.compare_fields(theta_2d_long,theta_2d_short,tDc_long,tDc_short,np.ones(nlong)/nlong,self.chom,shp=shp_2d)
                _,_,_,_,q_long_grid,q_short_grid = helper.compare_fields(theta_2d_long,theta_2d_short,q_long,q_short,np.ones(nlong)/nlong,self.chom,shp=shp_2d)
                _,_,_,_,w_long_grid,w_short_grid = helper.compare_fields(theta_2d_long,theta_2d_short,np.ones(nlong),np.ones(self.nshort),np.ones(nlong)/nlong,self.chom,shp=shp_2d,avg_flag=False)
                print("w_long_grid: min={}, max={}".format(np.min(w_long_grid),np.max(w_long_grid)))
                print("w_short_grid: min={}, max={}".format(np.min(w_short_grid),np.max(w_short_grid)))
                eps_grid = np.abs(q_short_grid - q_long_grid)
                eps_grid += 1e-3*(eps_grid > 0)*(eps_grid < 1e-3)
                print("eps_grid: min={}, max={}".format(np.nanmin(eps_grid),np.nanmax(eps_grid)))
                q_var_long_grid = q_long_grid*(1 - q_long_grid)
                print("q_var_long_grid: min={}, max={}".format(np.nanmin(q_var_long_grid),np.nanmax(q_var_long_grid)))
                q_var_short_grid = q_short_grid*(1 - q_short_grid)
                print("q_var_short_grid: min={}, max={}".format(np.nanmin(q_var_short_grid),np.nanmax(q_var_short_grid)))
                Teps_short = q_var_short_grid*(eps_grid!=0)/(eps_grid+1*(eps_grid==0))**2*tDc_short_grid
                print("Teps_short: min={}, max={}".format(np.nanmin(Teps_short),np.nanmax(Teps_short)))
                Teps_long = q_var_long_grid*(eps_grid!=0)/(eps_grid+1*(eps_grid==0))**2*tDc_long_grid
                print("Teps_long: min={}, max={}".format(np.nanmin(Teps_long),np.nanmax(Teps_long)))
                Ttot_short = np.nansum(Teps_short*w_short_grid)
                Ttot_long = np.nansum(Teps_long*w_long_grid)
                print("Ttot_short = {}".format(Ttot_short))
                print("Ttot_long = %f"%Ttot_long)
                #sys.exit()
                fig,ax = plt.subplots(ncols=2,figsize=(12,6))
                th01,th10 = np.meshgrid(theta_2d_units[0]*thaxes[0],theta_2d_units[1]*thaxes[1],indexing='ij')
                locator = ticker.LogLocator(numticks=6)
                im = ax[0].contourf(th01,th10,Teps_short.reshape(shp_2d),locator=locator,cmap=plt.cm.coolwarm)
                fig.colorbar(im,ax=ax[0],orientation='horizontal')
                ax[0].set_xlabel(theta_2d_names[0])
                ax[0].set_ylabel(theta_2d_names[1])
                ax[0].set_title("DGA total: %.3e"%Ttot_short)
                im = ax[1].contourf(th01,th10,Teps_long.reshape(shp_2d),locator=locator,cmap=plt.cm.coolwarm)
                fig.colorbar(im,ax=ax[1],orientation='horizontal')
                ax[1].set_xlabel(theta_2d_names[0])
                ax[1].set_ylabel(theta_2d_names[1])
                ax[1].set_title("EMP total: %.3e"%Ttot_long)
                fig.savefig(join(self.savefolder,fsuff+'_2d_Teps'))
                plt.close(fig)
                # ------------------
                # x->A
                if i == 0:
                    uname = r"$P\{x\to A\}$"
                else:
                    uname = r"$E[(%s)^{%d}1_A(X(\tau^+))]$"%(model.dam_dict[keys[k]]['name_fwd'],i)
                fsuff = model.dam_dict[keys[k]]['abb_fwd']+'%d_xa'%i
                logscale = model.dam_dict[keys[k]]['logscale']
                u0 = self.dam_moments[keys[k]]['xa'][i,:,0]
                u1 = self.dam_emp[keys[k]]['x_Dc']**i*(self.long_to_label==-1)
                fig,ax = helper.compare_plot_fields_2d(theta_2d_short,theta_2d_long,u0,u1,self.chom,np.ones(len(x_long)),theta_names=theta_2d_names,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_2d)
                fig.savefig(join(self.savefolder,fsuff+'_2d'))
                plt.close(fig)
                fig,ax = helper.compare_plot_fields_1d(theta_1d_short,theta_1d_long,u0,u1,self.chom,np.ones(nlong)/nlong,theta_name=theta_1d_name,theta_units=theta_1d_units,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_1d)
                fig.savefig(join(self.savefolder,fsuff+'_1d'))
                plt.close(fig)
                # ------------------
                # A->x
                if i == 0:
                    uname = r"$P\{A\to x\}$"
                else:
                    uname = r"$E[(%s)^{%d}1_A(X(\tau^-))]$"%(model.dam_dict[keys[k]]['name_bwd'],i)
                fsuff = model.dam_dict[keys[k]]['abb_bwd']+'%d_ax'%i
                logscale = model.dam_dict[keys[k]]['logscale']
                u0 = self.dam_moments[keys[k]]['ax'][i,:,0]
                u1 = self.dam_emp[keys[k]]['Dc_x']**i*(self.long_from_label==-1)
                fig,ax = helper.compare_plot_fields_2d(theta_2d_short,theta_2d_long,u0,u1,self.chom,np.ones(len(x_long)),theta_names=theta_2d_names,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_2d)
                fig.savefig(join(self.savefolder,fsuff+'_2d'))
                plt.close(fig)
                fig,ax = helper.compare_plot_fields_1d(theta_1d_short,theta_1d_long,u0,u1,self.chom,np.ones(nlong)/nlong,theta_name=theta_1d_name,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_1d)
                fig.savefig(join(self.savefolder,fsuff+'_1d'))
                plt.close(fig)
                # ------------------
                # B->x
                if i == 0:
                    uname = r"$P\{B\to x\}$"
                else:
                    uname = r"$E[(%s)^{%d}1_B(X(\tau^-))]$"%(model.dam_dict[keys[k]]['name_bwd'],i)
                fsuff = model.dam_dict[keys[k]]['abb_bwd']+'%d_bx'%i
                logscale = model.dam_dict[keys[k]]['logscale']
                u0 = self.dam_moments[keys[k]]['bx'][i,:,0]
                u1 = self.dam_emp[keys[k]]['Dc_x']**i*(self.long_from_label==1)
                fig,ax = helper.compare_plot_fields_2d(theta_2d_short,theta_2d_long,u0,u1,self.chom,np.ones(len(x_long)),theta_names=theta_2d_names,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_2d)
                fig.savefig(join(self.savefolder,fsuff+'_2d'))
                plt.close(fig)
                fig,ax = helper.compare_plot_fields_1d(theta_1d_short,theta_1d_long,u0,u1,self.chom,np.ones(nlong)/nlong,theta_name=theta_1d_name,u_names=[uname+" DGA",uname+" empirical"],avg_flag=True,logscale=logscale,shp=shp_1d)
                fig.savefig(join(self.savefolder,fsuff+'_1d'))
                plt.close(fig)
                # ------------------
            # Compare the field from A, from B, to A, to B, from A to B, and from B to A
            # Forward-in-time
        return
    def display_1d_densities_emp(self,model,data,theta_1d_abbs,theta_1d_orientations,fig=None,ax=None,include_reactive=True,phases=['aa','ab','bb','ba'],save_flag=False):
        funlib = model.observable_function_library()
        keys = list(model.dam_dict.keys())
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        comm_bwd = 1.0*(self.long_from_label==-1)
        comm_fwd = 1.0*(self.long_to_label==1)
        field = self.dam_moments[keys[0]]['ab'][0] # just (q-)*(q+)
        piab = comm_fwd*comm_bwd
        piab *= 1.0/np.mean(piab)
        piba = (1-comm_fwd)*(1-comm_bwd)
        piba *= 1.0/np.mean(piba)
        piaa = (1-comm_fwd)*comm_bwd
        piaa *= 1.0/np.mean(piaa)
        pibb = comm_fwd*(1-comm_bwd)
        pibb *= 1.0/np.mean(pibb)
        weight = np.ones(len(t_long))/len(t_long)
        for k in range(len(theta_1d_abbs)):
            fun0 = funlib[theta_1d_abbs[k]]
            def theta_1d_fun(x):
                th = fun0["fun"](x).reshape((len(x),1))
                return th
            theta_a,theta_b = theta_1d_fun(model.xst).flatten()
            theta_1d_name = fun0["name"]
            theta_1d_units = fun0["units"]
            theta_1d_unit_symbol = fun0["unit_symbol"]
            theta_x = theta_1d_fun(x_long)
            print("theta_x.shape = {}".format(theta_x.shape))
            if fig is None or ax is None:
                fig,ax = plt.subplots(figsize=(6,6))
            _,_,hpi = helper.plot_field_1d(theta_x,np.ones(len(t_long)),weight,avg_flag=False,color='black',label=r"$\pi$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,thetaname=theta_1d_name,density_flag=True,fig=fig,ax=ax,linewidth=2.5)
            handles = [hpi]
            if 'ab' in phases:
                _,_,hpiab = helper.plot_field_1d(theta_x,piab,weight,avg_flag=False,color='darkorange',label=r"$\pi_{AB}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpiab]
            if 'ba' in phases:
                _,_,hpiba = helper.plot_field_1d(theta_x,piba,weight,avg_flag=False,color='springgreen',label=r"$\pi_{BA}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpiba]
            if 'aa' in phases:
                _,_,hpiaa = helper.plot_field_1d(theta_x,piaa,weight,avg_flag=False,color='skyblue',label=r"$\pi_{AA}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpiaa]
            if 'bb' in phases:
                _,_,hpibb = helper.plot_field_1d(theta_x,pibb,weight,avg_flag=False,color='red',label=r"$\pi_{BB}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpibb]
            ax.legend(handles=handles,prop={'size': 25})
            ax.set_xlabel("DNS Probability density",fontdict=font)
            ax.tick_params(axis='both', which='major', labelsize=35)
            # Plot the lines for A and B
            print("Plotting lines for A and B")
            if theta_1d_orientations[k] == 'horizontal':
                ylim = ax.get_ylim()
                ax.plot(theta_a*theta_1d_units*np.ones(2),ylim,color='deepskyblue')
                ax.plot(theta_b*theta_1d_units*np.ones(2),ylim,color='red')
            elif theta_1d_orientations[k] == 'vertical':
                xlim = ax.get_xlim()
                ax.plot(xlim,theta_a*theta_1d_units*np.ones(2),color='deepskyblue')
                ax.plot(xlim,theta_b*theta_1d_units*np.ones(2),color='red')
            if save_flag:
                fig.savefig(join(self.savefolder,"dens1d_{}_emp".format(theta_1d_abbs[k])))
                plt.close(fig)
            plt.close(fig)
        return
    def display_1d_densities(self,model,data,theta_1d_abbs,theta_1d_orientations,fig=None,ax=None,include_reactive=True,phases=['aa','ab','bb','ba'],save_flag=False):
        funlib = model.observable_function_library()
        keys = list(model.dam_dict.keys())
        field = self.dam_moments[keys[0]]['ab'][0] # just (q-)*(q+)
        comm_bwd = self.dam_moments[keys[0]]['ax'][0,:,0]
        comm_fwd = self.dam_moments[keys[0]]['xb'][0,:,0]
        piab = comm_fwd*comm_bwd
        piab *= 1.0/np.sum(piab*self.chom)
        piba = (1-comm_fwd)*(1-comm_bwd)
        piba *= 1.0/np.sum(piba*self.chom)
        piaa = (1-comm_fwd)*comm_bwd
        piaa *= 1.0/np.sum(piaa*self.chom)
        pibb = comm_fwd*(1-comm_bwd)
        pibb *= 1.0/np.sum(pibb*self.chom)
        weight = self.chom
        for k in range(len(theta_1d_abbs)):
            fun0 = funlib[theta_1d_abbs[k]]
            def theta_1d_fun(x):
                th = fun0["fun"](x).reshape((len(x),1))
                return th
            theta_a,theta_b = theta_1d_fun(model.xst).flatten()
            theta_1d_name = fun0["name"]
            theta_1d_units = fun0["units"]
            theta_1d_unit_symbol = fun0["unit_symbol"]
            theta_x = theta_1d_fun(data.X[:,0])
            print("theta_x.shape = {}".format(theta_x.shape))
            if fig is None or ax is None:
                fig,ax = plt.subplots(figsize=(6,6))
            _,_,hpi = helper.plot_field_1d(theta_x,np.ones(data.nshort),weight,avg_flag=False,color='black',label=r"$\pi$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,thetaname=theta_1d_name,density_flag=True,fig=fig,ax=ax,linewidth=2.5)
            handles = [hpi]
            if 'ab' in phases:
                _,_,hpiab = helper.plot_field_1d(theta_x,piab,weight,avg_flag=False,color='darkorange',label=r"$\pi_{AB}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpiab]
            if 'ba' in phases:
                _,_,hpiba = helper.plot_field_1d(theta_x,piba,weight,avg_flag=False,color='springgreen',label=r"$\pi_{BA}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpiba]
            if 'aa' in phases:
                _,_,hpiaa = helper.plot_field_1d(theta_x,piaa,weight,avg_flag=False,color='skyblue',label=r"$\pi_{AA}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpiaa]
            if 'bb' in phases:
                _,_,hpibb = helper.plot_field_1d(theta_x,pibb,weight,avg_flag=False,color='red',label=r"$\pi_{BB}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpibb]
            ax.legend(handles=handles,prop={'size': 25})
            ax.set_xlabel("Probability density",fontdict=giantfont)
            ax.tick_params(axis='both', which='major', labelsize=35)
            # Plot the lines for A and B
            print("Plotting lines for A and B")
            if theta_1d_orientations[k] == 'horizontal':
                ylim = ax.get_ylim()
                ax.plot(theta_a*theta_1d_units*np.ones(2),ylim,color='deepskyblue')
                ax.plot(theta_b*theta_1d_units*np.ones(2),ylim,color='red')
            elif theta_1d_orientations[k] == 'vertical':
                xlim = ax.get_xlim()
                ax.plot(xlim,theta_a*theta_1d_units*np.ones(2),color='deepskyblue')
                ax.plot(xlim,theta_b*theta_1d_units*np.ones(2),color='red')
            if save_flag:
                fig.savefig(join(self.savefolder,"dens1d_{}".format(theta_1d_abbs[k])))
                plt.close(fig)
        return
    def display_2d_currents(self,model,data,theta_2d_abbs):
        funlib = model.observable_function_library()
        for k in range(len(theta_2d_abbs)):
            fun0 = funlib[theta_2d_abbs[k][0]]
            fun1 = funlib[theta_2d_abbs[k][1]]
            def theta_2d_fun(x):
                th = np.zeros((len(x),2))
                th[:,0] = fun0["fun"](x).flatten()
                th[:,1] = fun1["fun"](x).flatten()
                return th
            theta_2d_names = [fun0["name"],fun1["name"]] #[r"$|\Psi(30 km)|$",r"$U(30 km)$"]
            theta_2d_units = np.array([fun0["units"],fun1["units"]])
            theta_2d_unit_symbols = [fun0["unit_symbol"],fun1["unit_symbol"]]
            self.display_change_of_measure_current(model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs[k])
            self.display_dam_moments_abba_current(model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs[k],horz_lines=0)
        return
    def display_change_of_measure_current(self,model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs):
        # Put the equilibrium current on top of the change of measure
        Nx,Nt,xdim = data.X.shape
        shp_1d = np.array([20])
        shp_2d = np.array([20,40])
        theta_2d_short = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
        comm_fwd = np.ones((Nx,Nt))
        comm_bwd = np.ones((Nx,Nt))
        field = np.ones((Nx,Nt))
        fieldname = "Steady-state"
        weight = self.chom
        ss = np.random.choice(np.arange(Nx),10000,replace=False)
        # No current 
        fig,ax = self.plot_field_2d(model,data,field,weight,theta_2d_short,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=False,current_bdy_flag=False,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,cmap=plt.cm.YlOrBr,ss=ss)
        fig.savefig(join(self.savefolder,"pij_nocurrent_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # With current
        fig,ax = self.plot_field_2d(model,data,field,weight,theta_2d_short,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,current_bdy_flag=False,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,cmap=plt.cm.YlOrBr)
        fig.savefig(join(self.savefolder,"pij_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        return
    def display_change_of_measure_validation(self,model,data,theta_1d_fun,theta_2d_fun,theta_1d_name,theta_2d_names,theta_1d_units,theta_2d_units):
        # Display the change of measure from both DGA and simulation
        shp_1d = np.array([20])
        shp_2d = np.array([20,40])
        t_long,x_long = model.load_long_traj(self.long_simfolder)
        nlong = len(x_long)
        theta_1d_short = theta_1d_fun(data.X[:,0])
        theta_1d_long = theta_1d_fun(x_long)
        theta_2d_short = theta_2d_fun(data.X[:,0])
        theta_2d_long = theta_2d_fun(x_long)
        print("theta_2d_short.shape = {}".format(theta_2d_short.shape))
        print("theta_2d_long.shape = {}".format(theta_2d_long.shape))
        print("Displaying change of measure")
        nlong = theta_1d_long.shape[0]
        shp_2d = np.array([10,10])
        fig,ax = helper.compare_plot_fields_2d(theta_2d_short,theta_2d_long,np.ones(self.nshort),np.ones(nlong),self.chom,np.ones(nlong)/nlong,theta_names=theta_2d_names,u_names=[r"$\pi$ DGA",r"$\pi$ Empirical"],avg_flag=False,logscale=True,shp=shp_2d)
        fig.savefig(join(self.savefolder,"fidelity_pi_2D"))
        plt.close(fig)
        shp_1d = np.array([10])
        fig,ax = helper.compare_plot_fields_1d(theta_1d_short,theta_1d_long,np.ones(self.nshort),np.ones(nlong),self.chom,np.ones(nlong)/nlong,theta_name=theta_1d_name,u_names=[r"$\pi$ DGA",r"$\pi$ Empirical"],avg_flag=False,logscale=True,shp=shp_1d)
        fig.savefig(join(self.savefolder,"fidelity_pi_1D"))
        plt.close(fig)
        return
    def compute_change_of_measure(self,model,data,function):
        # Solve the invariant measure
        print("Beginning change of measure")
        xx = data.X[:,0,:]
        bdy_dist = lambda x: np.ones(len(x))
        function.fit_data(data.X[:,0],bdy_dist)
        self.chom = function.compute_stationary_density(data)
        self.chom *= 1.0/np.sum(self.chom)
        print("chom range = {},{}".format(np.min(self.chom),np.max(self.chom)))
        print("chom positivity fraction = {}".format(np.mean(self.chom>0)))
        self.chom = np.maximum(self.chom,1e-10)
        self.chom *= 1.0/np.sum(self.chom)
        return
    def project_field(self,field,weight,theta_x,shp=None,avg_flag=True,bounds=None):
        # Given a vector-valued observable function evaluation theta_x, find the mean 
        # and standard deviation of the field across remaining dimensions
        # Also return some integrated version of the standard deviation
        thdim = theta_x.shape[1]
        if shp is None: shp = 20*np.ones(thdim,dtype=int) # number of INTERIOR
        if bounds is None:
            bounds = np.array([np.min(theta_x,0)-1e-10,np.max(theta_x,0)+1e-10]).T
        cgrid,egrid,dth = helper.both_grids(bounds, shp+1)
        thaxes = [np.linspace(bounds[i,0]+dth[i]/2,bounds[i,1]-dth[i]/2,shp[i]) for i in range(thdim)]
        data_bins = ((theta_x - bounds[:,0])/dth).astype(int)
        for d in range(len(shp)):
            data_bins[:,d] = np.maximum(data_bins[:,d],0)
            data_bins[:,d] = np.minimum(data_bins[:,d],shp[d]-1)
        data_bins_flat = np.ravel_multi_index(data_bins.T,shp) # maps data points to bin
        Ncell = np.prod(shp)
        field_mean = np.nan*np.ones(Ncell)
        field_std = np.nan*np.ones(Ncell)
        for i in range(Ncell):
            idx = np.where(data_bins_flat == i)[0]
            if len(idx) > 0:
                weightsum = np.sum(weight[idx])
                field_mean[i] = np.sum(field[idx]*weight[idx])
                if avg_flag and (weightsum != 0):
                    field_mean[i] *= 1/weightsum
                    field_std[i] = np.sqrt(np.sum((field[idx]-field_mean[i])**2*weight[idx]))
                    field_std[i] *= 1/np.sqrt(weightsum)
        field_std_L2 = np.sqrt(np.nansum(field_std**2)/Ncell) #*np.prod(dth))
        field_std_Linf = np.nanmax(field_std)*np.prod(dth)
        return shp,dth,thaxes,cgrid,field_mean,field_std,field_std_L2,field_std_Linf,bounds
    def tendency_during_transition(self,model,data,theta_x,comm_bwd,comm_fwd):
        Nx,Nt,thdim = theta_x.shape
        xdim = data.X.shape[-1]
        bdy_dist = lambda x: (np.minimum(model.adist(x),model.bdist(x)))
        bdy_dist_x = bdy_dist(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
        # We need to have at least one point ahead and one point behind, for every point. 
        lag_time_max = self.lag_time_current_display
        data.insert_boundaries_fwd(bdy_dist_x,lag_time_max/2,lag_time_max)
        data.insert_boundaries_bwd(bdy_dist_x,lag_time_max/2,0)
        tidx = np.argmin(np.abs(int(lag_time_max/2) - data.t_x))
        comm_fwd_up = comm_fwd[np.arange(Nx),data.first_exit_idx_fwd]
        eps = 0.001
        comm_fwd_up[comm_fwd_up < eps] = np.nan
        comm_fwd_tidx = comm_fwd[:,tidx]
        comm_fwd_tidx[comm_fwd_tidx < eps] = np.nan
        comm_bwd_dn = comm_bwd[np.arange(Nx),data.first_exit_idx_bwd]
        comm_bwd_dn[comm_bwd_dn < eps] = np.nan
        comm_bwd_tidx = comm_bwd[:,tidx]
        comm_bwd_tidx[comm_bwd_tidx < eps] = np.nan
        Tup = (comm_fwd_up*theta_x[np.arange(Nx),data.first_exit_idx_fwd,:].T/comm_fwd_tidx).T
        Tdn = (comm_bwd_dn*theta_x[np.arange(Nx),data.first_exit_idx_bwd,:].T/comm_bwd_tidx).T
        T0 = theta_x[:,tidx]
        dt = data.t_x[data.first_exit_idx_fwd] - data.t_x[data.first_exit_idx_bwd]
        dt[dt == 0] = np.nan
        dtup = data.t_x[data.first_exit_idx_fwd] - data.t_x[tidx]
        dtup[dtup == 0] = np.nan
        dtdn = data.t_x[data.first_exit_idx_bwd] - data.t_x[tidx]
        dtdn[dtdn == 0] = np.nan
        L = ((Tup - T0).T/dt).T #((Tup - Tdn).T/dt).T
        return L,tidx
    def project_current_new(self,model,data,theta_x,comm_bwd,comm_fwd):
        # compute J_(AB)\cdot\nabla\theta. theta is a multi-dimensional observable, so we end up with a vector of that size.
        # This should be used hopefully for maximizing the reactive flux on a surface. 
        Nx,Nt,thdim = theta_x.shape
        Jtheta = np.zeros((Nx,thdim))
        bdy_dist = lambda x: np.minimum(model.adist(x),model.bdist(x))
        #data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current)
        data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current_display)
        Jtheta_up = ((self.chom*comm_bwd[:,0]*comm_fwd[np.arange(Nx),data.first_exit_idx]/data.t_x[data.first_exit_idx])*(theta_x[np.arange(Nx),data.first_exit_idx] - theta_x[:,0]).T).T
        Jtheta_dn = ((self.chom*comm_fwd[np.arange(Nx),data.last_idx]*comm_bwd[np.arange(Nx),data.last_entry_idx]/(data.t_x[data.last_idx] - data.t_x[data.last_entry_idx]))*(theta_x[np.arange(Nx),data.last_idx] - theta_x[np.arange(Nx),data.last_entry_idx]).T).T
        return Jtheta_up,Jtheta_dn
    def project_current(self,data,theta_x,theta_yj,theta_xpj,theta_ypj,shp,comm_bwd,comm_fwd,ss=None):
        # data must have already the current boundaries inserted
        # Use the project_field utility in the data object
        # If dirn=1, current from A to B. Otherwise, current from B to A
        if ss is None: ss = np.arange(self.nshort)
        chomss = self.chom[ss]/np.sum(self.chom[ss])
        shp = np.array(shp)
        #if dirn != 0:
        comm_bwd_x = comm_bwd[ss,0] #self.phi_x_bound.dot(comm_bwd_phi)
        comm_bwd_xpj = comm_bwd[ss,data.last_entry_idx[ss]] #self.phi_xpj_bound.dot(comm_bwd_phi)
        comm_fwd_yj = comm_fwd[ss,data.last_idx[ss]] #self.phi_yj_bound.dot(comm_fwd_phi)
        comm_fwd_ypj = comm_fwd[ss,data.first_exit_idx[ss]] #self.phi_ypj_bound.dot(comm_fwd_phi)
        print("chomss.shape = {}".format(chomss.shape))
        print("comm_bwd_x.shape = {}".format(comm_bwd_x.shape))
        print("comm_fwd_ypj.shape = {}".format(comm_fwd_ypj.shape))
        print("theta_ypj.shape = {}".format(theta_ypj.shape))
        print("theta_x.shape = {}".format(theta_x.shape))
        field0 = (chomss*comm_bwd_x*comm_fwd_ypj*(theta_ypj[ss] - theta_x[ss]).T).T
        field1 = (chomss*comm_bwd_xpj*comm_fwd_yj*(theta_yj[ss] - theta_xpj[ss]).T).T
        print("field0.shape = {}, field1.shape = {}".format(field0.shape,field1.shape))
        thdim = field0.shape[1]
        if shp is None: shp = 20*np.ones(thdim,dtype=int) # number of INTERIOR
        Ncell = np.prod(shp)
        J = np.zeros((Ncell,thdim))
        for d in range(thdim):
            _,dth,thaxes,cgrid,J0,J0_std,_,_,_ = helper.project_field(field0[:,d],chomss,theta_x[ss],shp=shp,avg_flag=False)
            _,dth,thaxes,cgrid,J1,J1_std,_,_,_ = helper.project_field(field1[:,d],chomss,theta_yj[ss],shp=shp,avg_flag=False)
            J[:,d] = 1/(2*self.lag_time_current_display*np.prod(dth))*(J0 + J1)
        return thaxes,J
    def plot_field_2d(self,model,data,field,weight,theta_x,shp=[60,60],cmap=plt.cm.coolwarm,fieldname="",fun0name="",fun1name="",current_flag=False,current_bdy_flag=False,comm_bwd=None,comm_fwd=None,current_shp=[25,25],abpoints_flag=False,theta_ab=None,avg_flag=True,logscale=False,ss=None,magu_fw=None,magu_obs=None,units=np.ones(2),unit_symbols=["",""],cbar_orientation='horizontal',fig=None,ax=None,vmin=None,vmax=None,contourf_flag=True,contour_notf_flag=False,contour_notf_levels=None):
        fig,ax = helper.plot_field_2d(field[:,0],weight,theta_x[:,0],shp=shp,cmap=cmap,fieldname=fieldname,fun0name=fun0name,fun1name=fun1name,avg_flag=avg_flag,std_flag=False,logscale=logscale,ss=ss,units=units,unit_symbols=unit_symbols,cbar_orientation=cbar_orientation,fig=fig,ax=ax,vmin=vmin,vmax=vmax,contourf_flag=contourf_flag,contour_notf_flag=contour_notf_flag,contour_notf_levels=contour_notf_levels)
        if abpoints_flag:
            ass_theta = self.aidx
            bss_theta = self.bidx
            ax.scatter(units[0]*theta_x[ass_theta,0,0],units[1]*theta_x[ass_theta,0,1],color='lightgray',marker='.',zorder=2)
            ax.scatter(units[0]*theta_x[bss_theta,0,0],units[1]*theta_x[bss_theta,0,1],color='lightgray',marker='.',zorder=2)
            print("ABpoints done")
        if theta_ab is not None:
            ax.text(units[0]*theta_ab[0,0],units[1]*theta_ab[0,1],asymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=15,horizontalalignment='center',verticalalignment='center',zorder=100)
            ax.text(units[0]*theta_ab[1,0],units[1]*theta_ab[1,1],bsymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=15,horizontalalignment='center',verticalalignment='center',zorder=100)
        if current_flag:
            if current_bdy_flag: # The equilibrium current will not have this.
                bdy_dist = lambda x: np.minimum(model.adist(x),model.bdist(x))
            else:
                bdy_dist = lambda x: np.ones(x.shape[0])
            data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current_display)
            #theta_yj,theta_xpj,theta_ypj = thetaj # thetaj better not be None either
            print("comm_bwd.shape = {}".format(comm_bwd.shape))
            print("comm_fwd.shape = {}".format(comm_fwd.shape))
            print("About to go into plotting current")
            thaxes_current,J = self.project_current(data,theta_x[:,0],theta_x[np.arange(data.nshort),data.last_idx],theta_x[np.arange(data.nshort),data.last_entry_idx],theta_x[np.arange(data.nshort),data.first_exit_idx],current_shp,comm_bwd,comm_fwd,ss=ss)
            dth = np.array([thax[1] - thax[0] for thax in thaxes_current])
            Jmag_full = np.sqrt(np.sum(J**2, 1))
            minmag,maxmag = np.nanmin(Jmag_full),np.nanmax(Jmag_full)
            print("minmag={}, maxmag={}".format(minmag,maxmag))
            th0_subset = np.arange(current_shp[0]) #np.linspace(0,p[0]-2,current_shp[0]-2)).astype(int)
            th1_subset = np.arange(current_shp[1]) #np.linspace(0,len(thaxes_current[1])-2,min(shp[1]-2,current_shp[1]-2)).astype(int)
            J0 = J[:,0].reshape(current_shp)[th0_subset,:][:,th1_subset] #*units[0]
            J1 = J[:,1].reshape(current_shp)[th0_subset,:][:,th1_subset] #*units[1]
            Jmag = np.sqrt(J0**2 + J1**2)
            print("Jmag range = ({},{})".format(np.nanmin(Jmag),np.nanmax(Jmag)))
            print("J0.shape = {}".format(J0.shape))
            dsmin,dsmax = 0*np.max(current_shp)/40,np.max(current_shp)/20 # lengths if arrows in grid box units
            coeff1 = 10.0/maxmag
            coeff0 = dsmax / (np.exp(-coeff1 * maxmag) - 1)
            ds = coeff0 * (np.exp(-coeff1 * Jmag) - 1)
            #ds = dsmin + (dsmax - dsmin)*(Jmag - minmag)/(maxmag - minmag)
            normalizer = ds*(Jmag != 0)/(np.sqrt((J0/(dth[0]))**2 + (J1/(dth[1]))**2) + (Jmag == 0))
            J0 *= normalizer*(1 - np.isnan(J0))
            J1 *= normalizer*(1 - np.isnan(J1))
            print("Final J0 range for ({}, {}) = ({},{})".format(fun0name,fun1name,np.nanmin(J0),np.nanmax(J0)))
            th01_subset,th10_subset = np.meshgrid(units[0]*thaxes_current[0][th0_subset],units[1]*thaxes_current[1][th1_subset],indexing='ij')
            ax.quiver(th01_subset,th10_subset,units[0]*J0,units[1]*J1,angles='xy',scale_units='xy',scale=1.0,color='black',width=1.4,headwidth=4.0,units='dots',zorder=4) # was width=2.0, headwidth=2.7
            print(f"Quivering done for ({fun0name},{fun1name})")
        if magu_obs is not None:
            print(f"Beginning to plot magu_obs")
            for ti in range(len(magu_obs)):
                ax.plot(magu_obs[ti][:,0]*units[0],magu_obs[ti][:,1]*units[1],color='deepskyblue',zorder=3,alpha=1.0,linestyle='solid',linewidth=0.85)
            print(f"Finishing plot of magu_obs")
        if magu_fw is not None:
            ax.plot(magu_fw[:,0]*units[0],magu_fw[:,1]*units[1],color='cyan',linewidth=2.0,zorder=5,linestyle='solid')
        print("Returning from function plot_field_2d")
        return fig,ax 
    def inverse_committor_slice(self,field,comm_levels):
        comm_fwd = self.dam_moments['one']['xb'][0]
        comm_bwd = self.dam_moments['one']['ax'][0]
        # Try making comm_bwd just all ones
        #comm_bwd[:] = 1.0
        print("field range: {},{}".format(np.nanmin(field),np.nanmax(field)))
        weight = np.maximum(self.chom,0)
        good_idx = np.where(np.isnan(field)==0)[0]
        shp,dth,thaxes,cgrid,q_mean,q_std,_,_,_ = self.project_field(comm_fwd[good_idx,0],weight[good_idx],field[good_idx].reshape((len(good_idx),1)))
        print("thaxes[0] = {}".format(thaxes[0]))
        print("q_mean = {}".format(q_mean))
        field_levels = np.zeros(len(comm_levels))
        qab_levels = np.zeros(len(comm_levels))
        for i in range(len(field_levels)):
            print("q_mean range = ({},{})".format(np.nanmin(q_mean),np.nanmax(q_mean)))
            thidx0 = np.where(np.abs(np.diff(np.sign(q_mean - comm_levels[i]))) == 2)[0]
            if len(thidx0) > 0:
                print("thidx0 = {}".format(thidx0))
                thidx0 = thidx0[0]
                thidx1 = thidx0 + 1
                th0,th1 = thaxes[0][thidx0],thaxes[0][thidx1]
                q0,q1 = q_mean[thidx0],q_mean[thidx1]
                print("q0={}, q1={}".format(q0,q1))
                field_levels[i] = th0 + (comm_levels[i] - q0)*(th1 - th0)/(q1 - q0)
                # Now take the average of (qp**2*qm) and (qp*qm) over all points with th0 <= field <= th1
                data_idx = np.where((field>=th0)*(field<=th1))[0]
                qab_levels[i] = np.sum(comm_fwd[data_idx,0]**2*comm_bwd[data_idx,0]*self.chom[data_idx])/np.sum(comm_fwd[data_idx,0]*comm_bwd[data_idx,0]*self.chom[data_idx])
            else:
                field_levels[i] = thaxes[0][np.argmin(np.abs(q_mean-comm_levels[i]))]
                qab_levels[i] = np.nan
        return field_levels,qab_levels
    def compare_lead_times(self,weight,field,shp=[20,],fieldname="",avg_flag=True,std_flag=False,fig=None,ax=None,color='green',label="",linestyle='-',linewidth=1,orientation='horizontal',units=1.0,unit_symbol=""):
        comm_fwd = self.dam_moments['one']['xb'][0]
        qlevels = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        field_slices = self.inverse_committor_slice(field,qlevels)
        tq = np.zeros(len(qlevels))
        tf = np.zeros(len(qlevels))
        for i in range(len(qlevels)):
            qidx = np.where(np.abs((comm_fwd[:,0]-qlevels[i])) < 0.025)[0]
            fidx = np.where(np.abs((field - field_slices[i])/np.max(field_slices)) < 0.025)[0]
            tq[i] = np.sum(self.mfpt_b_ava_fwd[qidx,0]*self.chom[qidx])/np.sum(self.chom[qidx])
            tf[i] = np.sum(self.mfpt_b_ava_fwd[fidx,0]*self.chom[fidx])/np.sum(self.chom[fidx])
        fig,ax = plt.subplots()
        ax.set_xlim([0,max(np.max(tf),np.max(tq))])
        ax.set_ylim([0,max(np.max(tf),np.max(tq))])
        ax.plot(tf,tq,color='black',marker='o')
        ax.plot(ax.get_xlim(),ax.get_ylim(),linestyle='--',color='black')
        ax.set_xlabel(r"$E[\tau_B|\tau_B<\tau_A; %s]$"%fieldname)
        ax.set_ylabel(r"$E[\tau_B|\tau_B<\tau_A; q^+]$")
        return fig,ax
    def plot_field_1d(self,field,weight,theta_x,shp=[20,],fieldname="",funname="",avg_flag=True,std_flag=False,fig=None,ax=None,color='green',label="",linestyle='-',linewidth=1,orientation='horizontal',units=1.0,unit_symbol=""):
        shp = np.array(shp)
        # Plot a 1d scatterplot of a field average across remaining dimensions
        shp,dth,thaxes,cgrid,field_mean,field_std,field_std_L2,field_std_Linf,_ = helper.project_field(field,weight,theta_x.reshape((len(theta_x),1)),shp,avg_flag=avg_flag)
        print("shp0 = {}, dth={}".format(shp,dth*units))
        print("thaxes in ({},{})".format(thaxes[0][0]*units,thaxes[0][-1]*units))
        print("field in ({},{}), field_mean in ({},{})".format(np.nanmin(field),np.nanmax(field),np.nanmin(field_mean),np.nanmax(field_mean)))
        if (fig is None) or (ax is None):
            fig,ax = plt.subplots(figsize=(6,6))
        if orientation=='horizontal':
            handle, = ax.plot(units*thaxes[0],field_mean,marker='o',linestyle=linestyle,color=color,label=label,linewidth=linewidth)
            if std_flag:
                ax.plot(units*thaxes[0],field_mean-field_std,color=color,linestyle='--',linewidth=linewidth)
                ax.plot(units*thaxes[0],field_mean+field_std,color=color,linestyle='--',linewidth=linewidth)
            xlab = funname
            if len(unit_symbol) > 0: xlab += " ({})".format(unit_symbol)
            ax.set_xlabel(xlab,fontdict=font)
            ax.set_ylabel(fieldname,fontdict=font)
        else:
            handle, = ax.plot(field_mean,units*thaxes[0],marker='o',linestyle=linestyle,color=color,label=label,linewidth=linewidth)
            if std_flag:
                ax.plot(field_mean-field_std,units*thaxes[0],color=color,linestyle='--')
                ax.plot(field_mean+field_std,units*thaxes[0],color=color,linestyle='--')
            ylab = funname
            if len(unit_symbol) > 0: ylab += " ({})".format(unit_symbol)
            ax.set_ylabel(ylab,fontdict=font)
            ax.set_xlabel(fieldname,fontdict=font)
        ax.tick_params(axis='x',labelsize=20)
        ax.tick_params(axis='y',labelsize=20)
        ax.set_xlim([units*(thaxes[0][0]-dth[0]/2),units*(thaxes[0][-1]+dth[-1]/2)])
        return fig,ax,handle
    def plot_zfam_committor(self,model,data):
        q = model.q
        funlib = model.observable_function_library()
        ss = np.random.choice(np.arange(self.nshort),size=min(self.nshort,100000),replace=False)
        # Transition states (only if we're seeing the full-state observable)
        z_list = q['z_d'][2:-1]/1000
        #z_list = np.append(z_list,-5)
        #fun_name_list = ['LASSO','U','mag','vTint','vT','dqdy','vq','q2'] #'LASSO']
        fun_name_list = ['U','mag','q2','dqdy','vq','LASSO']
        Nth = len(fun_name_list)
        #fig_list = []
        #ax_list = []
        fig,axes = plt.subplots(nrows=Nth,ncols=3,figsize=(3*6,Nth*6),constrained_layout=True,sharey='row')
        std_range = np.array([0,0.5])
        L2_range = np.array([np.inf,-np.inf])
        print("Beginning std_range = {}".format(std_range))
        for fi in range(len(fun_name_list)): #funlib.keys():
            fun_name = fun_name_list[fi]
            if fun_name == 'LASSO':
                fun_tex = 'LASSO'
                units = 1.0
                unit_symbol = ""
            else:
                fun_tex = funlib[fun_name]["name"]
                units = funlib[fun_name]["units"]
                unit_symbol = funlib[fun_name]["unit_symbol"]
            #theta_fun_list = [[] for i in range(len(z_list))]
            theta_fun_list = [] 
            for i in range(len(z_list)): #-1):
                if fun_name == 'LASSO':
                    def tfl(xx):
                        return (self.lasso_predict_onez(model,xx)).reshape((len(xx),q['Nz']-1))
                    theta_fun_list += [model.fun_at_level(tfl,z_list[i])]
                else:
                    theta_fun_list += [model.fun_at_level(funlib[fun_name]["fun"],z_list[i])]
            #if fun_name == 'LASSO':
            #    def tfl(xx):
            #        return (self.lasso_predict_allz(model,xx)).reshape((len(xx),1))
            #    theta_fun_list += [tfl]
            #else:
            #    theta_fun_list += [model.fun_zmean(funlib[fun_name]["fun"])]
            fieldname = r"$q^+$"
            funname = fun_tex #funlib[fun_name]["name"]
            print("Plotting zfam {}".format(funname))
            comm_fwd = self.dam_moments['one']['xb'][0]
            _,_,std_range_i,L2_range_i = self.plot_field_family_1d(data.X[ss,0],comm_fwd[ss,0],self.chom[ss]/np.sum(self.chom[ss]),theta_fun_list,z_list,fieldname=fieldname,funname=funname,cmap0=plt.cm.coolwarm,std_range=std_range,units=units,unit_symbol=unit_symbol,fig=fig,ax=axes[fi],cbar_flag=(fi==Nth-1))
            if fun_name in ['vT','mag','dqdy','q2']:
                for axi in range(2):
                    axes[fi,axi].xaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt_short))
                    axes[fi,axi].xaxis.set_major_locator(ticker.LinearLocator(numticks=3))
                    #ax[axi].locator_params(tight=True, nbins=3)

            std_range[0] = min(std_range[0],std_range_i[0])
            std_range[1] = max(std_range[1],std_range_i[1])
            L2_range[0] = min(L2_range[0],L2_range_i[0])
            L2_range[1] = max(L2_range[1],L2_range_i[1])
            #fig_list += [fig]
            #ax_list += [ax]
        print("Ending std_range = {}".format(std_range))
        L2_range_mag = L2_range[1] - L2_range[0]
        L2_range[0] -= 0.1*L2_range_mag
        L2_range[1] += 0.1*L2_range_mag
        for i in range(Nth):
            #ax_list[i][2].set_xlim(L2_range)
            axes[i,2].set_xlim(L2_range)
            #fig_list[i].subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9)
            #fig_list[i].savefig(join(self.savefolder,"qp_zfam_{}".format(fun_name_list[i])),bbox_inches="tight",pad_inches=0.2)
            #plt.close(fig_list[i])
        fig.savefig(join(self.savefolder,"zfam"),bbox_inches="tight",pad_inches=0.2)
        return
    def plot_field_family_1d(self,X,field,weight,theta_fun_list,z_list,shp=[40,],fieldname="",funname="",avg_flag=True,cmap0=plt.cm.get_cmap('coolwarm'),cmap1=plt.cm.get_cmap('magma'),std_range=None,units=1.0,unit_symbol="",fig=None,ax=None,cbar_flag=True):
        shp = np.array(shp).astype(int)
        n = len(z_list)
        if fig is None or ax is None:
            fig,ax = plt.subplots(ncols=3,figsize=(18,6),sharey=True,constrained_layout=True)
        field_mean = np.zeros((n,shp[0]))
        field_std = np.zeros((n,shp[0]))
        field_std_L2 = np.zeros(n)
        field_std_Linf = np.zeros(n)
        thaxes_list = np.zeros((n,shp[0]))
        for zi in range(n):
            #print("In LASSO loop. zi = {}".format(zi))
            theta_x = theta_fun_list[zi](X)
            shp,dth,thaxes,cgrid,field_mean[zi],field_std[zi],field_std_L2[zi],field_std_Linf[zi],_ = self.project_field(field,weight,theta_x,shp,avg_flag=avg_flag)
            thaxes_list[zi] = units*thaxes[0]
        if std_range is None: std_range = np.array([np.nanmin(field_std),np.nanmax(field_std)])
        L2_range = np.array([np.min(field_std_L2),np.max(field_std_L2)])
        for zi in range(n):
            sc_mean = ax[0].scatter(thaxes_list[zi],z_list[zi]*np.ones(shp[0]),c=field_mean[zi],cmap=cmap0,vmin=np.nanmin(field_mean),vmax=np.nanmax(field_mean),alpha=0.6,s=85.0)
            sc_std = ax[1].scatter(thaxes_list[zi],z_list[zi]*np.ones(shp[0]),c=field_std[zi],cmap=cmap1,vmin=std_range[0],vmax=std_range[1],alpha=0.6,s=85)
        ax[2].plot(field_std_L2[:].flatten(),z_list[:],color='black',marker='o')
        #ax[2].plot(field_std_L2[-1],z_list[-1],color='black',marker='o',markersize=20)
        if cbar_flag:
            cbar = fig.colorbar(sc_mean,ax=ax[0],orientation='horizontal')
            cbar.ax.tick_params(labelsize=30)
            cbar = fig.colorbar(sc_std,ax=ax[1],orientation='horizontal')
            cbar.ax.tick_params(labelsize=30)
        #ax[0].set_title("{} on {} ({})".format(fieldname,funname,unit_symbol),fontdict=bigfont)
        xlab = funname
        if len(unit_symbol) > 0: xlab += " (%s)"%(unit_symbol)
        ax[0].set_xlabel(xlab,fontdict=bigfont)
        ax[0].set_ylabel("z (km)",fontdict=bigfont)
        ax[0].tick_params(axis='x',labelsize=28)
        ax[0].tick_params(axis='y',labelsize=28)
        #ax[1].set_title("{} std. on {}".format(fieldname,funname),fontdict=bigfont)
        ax[1].set_xlabel(xlab,fontdict=bigfont)
        ax[1].set_ylabel("z (km)",fontdict=bigfont)
        ax[1].tick_params(axis='x',labelsize=28)
        ax[1].tick_params(axis='y',labelsize=28)
        ax[2].set_ylabel("z (km)",fontdict=bigfont)
        #ax[2].set_title(r"$L^2$ proj. error",fontdict=bigfont)
        ax[2].tick_params(axis='x',labelsize=28)
        ax[2].tick_params(axis='y',labelsize=28)
        ax[2].set_xlabel(r"Std$(q^+)$ over %s"%(funname),fontdict=bigfont)
        return fig,ax,std_range,L2_range
    def weighted_least_squares(self,x0,x1,weights):
        # Linear regression for two variables
        idx = np.where((np.isnan(x0)==0)*(np.isnan(x1)==0)*(np.isnan(weights)==0))[0]
        N = len(idx)
        X = np.array([np.ones(N),x0[idx]]).T
        xwx = (X.T*weights[idx]).dot(X)
        xwy = (X.T*weights[idx]).dot(x1[idx])
        coeffs = np.linalg.solve(xwx,xwy)
        return coeffs
    def out_of_sample_extension(self,field,data,xnew,ss_size=100000,k=15,inverse_lengthscale=1.0):
        # For a new sample (such as a long trajectory), extend the field to the new ones just by nearest-neighbor averaging
        # Update: find the nearest neighbor whose value is not NaN
        prng = np.random.RandomState(0)
        good_idx = np.where(np.isnan(field)==0)[0]
        #ss = np.random.choice(np.arange(self.nshort),size=min(self.nshort,100000),replace=False)
        ss = prng.choice(good_idx,size=min(len(good_idx),ss_size),replace=False)
        Xsq = np.sum(data.X[ss,0]**2,1)
        dsq = np.add.outer(Xsq,np.sum(xnew**2,1)) - 2*data.X[ss,0].dot(xnew.T)
        knn = np.argpartition(dsq,k+1,axis=0)
        knn = knn[:k,:]
        close_dsq = np.zeros((k,len(xnew)))
        for j in range(k):
            close_dsq[j] = dsq[knn[j],np.arange(len(xnew))]
        dsq = dsq[knn,np.arange(len(xnew))]
        close_field = field[ss[knn]]
        if np.any(np.isnan(close_field)):
            raise Exception("The close_field has some nan's, but we were supposed to discard those earlier")
        weights = np.exp(-inverse_lengthscale**2*close_dsq)
        weights = weights/np.sum(weights*(np.isnan(close_field)==0),0)
        fnew = np.nansum(close_field*weights,0)
        frange = np.nanmax(field) - np.nanmin(field)
        if np.nanmax(fnew) > np.nanmax(field) + 0.05*frange:
            raise Exception("ERROR: in out-of-sample-extension, nanmax(fnew) = {} while nanmax(field) = {}".format(np.nanmax(fnew),np.nanmax(field)))
        if np.nanmin(fnew) < np.nanmin(field) - 0.05*frange:
            raise Exception("ERROR: in out-of-sample-extension, nanmin(fnew) = {} while nanmin(field) = {}".format(np.nanmin(fnew),np.nanmin(field)))
        return fnew
    def plot_prediction_curves_colored(self,model,data):
        # Plot the prediction curves to B, colored by the two canonical coordinates
        eps = 1e-2
        qb = self.dam_moments['one']['xb'][0,:,0]
        qma = self.dam_moments['one']['ax'][0,:,0]
        piab = self.chom #*qb*qma
        piab = np.maximum(piab,0)
        piab *= 1.0/np.sum(piab)
        tb = self.dam_moments['one']['xb'][1,:,0]*(qb > eps)/(qb + 1*(qb < eps))
        tb[np.where(qb == 0)[0]] = np.nan
        qa = self.dam_moments['one']['xa'][0,:,0]
        qmb = self.dam_moments['one']['bx'][0,:,0]
        piba = self.chom #*qa*qmb
        piba = np.maximum(piba,0)
        piba *= 1.0/np.sum(piba)
        ta = self.dam_moments['one']['xa'][1,:,0]*(qa > eps)/(qa + 1*(qa < eps))
        ymin = np.nanquantile(np.concatenate((tb,ta)),0.05)
        ymax = np.nanquantile(np.concatenate((tb,ta)),0.99)
        xmin = 0.0
        xmax = 1.0
        theta_ab = np.array([[0.0,ymax+0.05*(ymax-ymin)],[1.0,0.0]])
        print("ymin = {}, ymax = {}".format(ymin,ymax))
        print("A avoiding B in range {},{}".format(np.nanmin(ta),np.nanmax(ta)))
        # x -> B
        funlib = model.observable_function_library()
        ss_xb = np.random.choice(np.arange(self.nshort),size=min(self.nshort,30000),replace=True,p=piab)
        Uref_ss_xb = funlib["Uref"]["fun"](data.X[ss_xb,0])*funlib["Uref"]["units"]
        vTintref_ss_xb = funlib["vTintref"]["fun"](data.X[ss_xb,0])*funlib["vTintref"]["units"]
        nn = np.where(np.isnan(tb[ss_xb])==0)[0]
        # ------------------------------
        # Invariant density
        fig,ax = plt.subplots(figsize=(6,6)) #,constrained_layout=True)
        theta_x = np.array([qb[ss_xb[nn]],tb[ss_xb[nn]]]).T
        ax.text(theta_ab[0,0],theta_ab[0,1],asymb,fontdict=font,bbox=dict(facecolor='white',alpha=1.0),color='black',horizontalalignment='center',verticalalignment='center',zorder=10)
        ax.text(theta_ab[1,0],theta_ab[1,1],bsymb,fontdict=font,bbox=dict(facecolor='white',alpha=1.0),color='black',horizontalalignment='center',verticalalignment='center',zorder=10)
        _,_ = helper.plot_field_2d(np.ones(len(nn)),piab[ss_xb[nn]],theta_x,avg_flag=False,fieldname=r"Density $\pi(x)$",fun0name=r"$q^+(x)$",fun1name=r"$\eta^+(x)$ (days)",fig=fig,ax=ax,std_flag=False,cbar_pad=0.05,cbar_location='top',shp=[15,15],cmap=plt.cm.binary,logscale=True)
        ylim = [ymin - 0.03*(ymax - ymin), ymax + 0.05*(ymax - ymin)]
        xlim = [xmin, xmax]
        #ax.set_ylim(ylim)
        #ax.set_xlim(xlim)
        #ax.set_ylabel(r"$E[\tau_B|x\to B]$ (days)",fontdict=font)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.tick_params(axis='both', which='major', labelsize=15)
        coeffs = self.weighted_least_squares(qb,tb,self.chom)
        print("prediction curves coeffs = {}".format(coeffs))
        x = np.array([np.min(qb),np.max(qb)])
        symbol = "+" if coeffs[1]>0 else ""
        #handle, = ax[0].plot(x,coeffs[0]+coeffs[1]*x,color='black',linewidth=3,label=r"$%.1f%s%.1fP\{x\to B\}$" % (coeffs[0],symbol,coeffs[1]))
        #ax[0].legend(handles=[handle],prop={'size': 20})
        #ax.set_title(r"Density $\pi$",fontdict=font)
        fig.savefig(join(self.savefolder,"qp_tb_coords_pi"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        # ------------------------------
        U_title = r"%s (%s)"%(funlib["Uref"]["name"],funlib["Uref"]["unit_symbol"])
        vT_title = r"%s (%s)"%(funlib["vTintref"]["name"],funlib["vTintref"]["unit_symbol"])
        # Zonal wind
        fig,ax = plt.subplots(ncols=2,figsize=(12,6),sharex=True,sharey=True) #,constrained_layout=True)
        #fig,ax0 = plt.subplots()
        ax0,ax1 = ax
        theta_x = np.array([qb[ss_xb[nn]],tb[ss_xb[nn]]]).T
        _,_ = helper.plot_field_2d(Uref_ss_xb[nn],piab[ss_xb[nn]],theta_x,fieldname=U_title,fun0name=r"$q^+(x)$",fun1name=r"$\eta^+(x)$",fig=fig,ax=ax0,std_flag=False,cbar_pad=0.05,cbar_location='top',shp=[15,15],cmap=plt.cm.plasma)
        ax0.text(theta_ab[0,0],theta_ab[0,1],asymb,fontdict=font,bbox=dict(facecolor='white',alpha=1.0),color='black',horizontalalignment='center',verticalalignment='center',zorder=10)
        ax0.text(theta_ab[1,0],theta_ab[1,1],bsymb,fontdict=font,bbox=dict(facecolor='white',alpha=1.0),color='black',horizontalalignment='center',verticalalignment='center',zorder=10)
        #ax0.set_ylim(ylim)
        #ax0.set_xlim(xlim)
        ax0.set_ylabel(r"$\eta^+(x)$ (days)",fontdict=font)
        #ax0.xaxis.set_major_locator(plt.MaxNLocator(nbins=2))
        #ax0.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax0.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax0.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax0.tick_params(axis='both', which='major', labelsize=15)
        coeffs = self.weighted_least_squares(qb,tb,self.chom)
        print("prediction curves coeffs = {}".format(coeffs))
        x = np.array([np.min(qb),np.max(qb)])
        symbol = "+" if coeffs[1]>0 else ""
        #handle, = ax[0].plot(x,coeffs[0]+coeffs[1]*x,color='black',linewidth=3,label=r"$%.1f%s%.1fP\{x\to B\}$" % (coeffs[0],symbol,coeffs[1]))
        #ax[0].legend(handles=[handle],prop={'size': 20})
        #ax0.set_title(r"%s (%s)"%(funlib["Uref"]["name"],funlib["Uref"]["unit_symbol"]),fontdict=font)
        # --------------------------------------
        # IHF
        theta_x = np.array([qb[ss_xb[nn]],tb[ss_xb[nn]]]).T
        ax1.text(theta_ab[0,0],theta_ab[0,1],asymb,fontdict=font,bbox=dict(facecolor='white',alpha=1.0),color='black',horizontalalignment='center',verticalalignment='center',zorder=10)
        ax1.text(theta_ab[1,0],theta_ab[1,1],bsymb,fontdict=font,bbox=dict(facecolor='white',alpha=1.0),color='black',horizontalalignment='center',verticalalignment='center',zorder=10)
        _,_ = helper.plot_field_2d(vTintref_ss_xb[nn],piab[ss_xb[nn]],theta_x,fieldname=vT_title,fun0name=r"$q^+(x)$",fun1name=r"$\eta^+(x)$",fig=fig,ax=ax1,cbar_pad=0.05,cbar_location='top',std_flag=False,shp=[15,15],cmap=plt.cm.plasma)
        #ax1.set_ylim(ylim)
        #ax1.set_xlim(xlim)
        ax1.set_xlabel(r"$q^+(x)$",fontdict=font)
        ax1.set_ylabel(r"$\eta^+(x)$ (days)",fontdict=font)
        ax1.yaxis.set_visible(False)
        #ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=2))
        ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        coeffs = self.weighted_least_squares(qb,tb,self.chom)
        print("prediction curves coeffs = {}".format(coeffs))
        x = np.array([np.min(qb),np.max(qb)])
        symbol = "+" if coeffs[1]>0 else ""
        ax1.tick_params(axis='both',which='major',labelsize=15)
        fig.savefig(join(self.savefolder,"qp_tb_coords"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        return
    def maximize_rflux_on_surface(self,model,data,theta_x,comm_bwd,comm_fwd,weight,theta_level,theta_tol,max_num_states,frac_of_max):
        print("theta_x.shape = {}".format(theta_x.shape))
        Jup,Jdn = self.project_current_new(model,data,theta_x,comm_bwd,comm_fwd)
        print("Jup.shape = {}, Jdn.shape = {}".format(Jup.shape,Jdn.shape))
        print("theta_level = {}".format(theta_level))
        Jup = Jup.flatten()
        Jdn = Jdn.flatten()
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
        nnidx = np.where(np.isnan(theta_x[:,tidx,0]) + np.isnan(Jup) + np.isnan(Jdn) == 0)[0]
        theta_discrepancy = np.sqrt(np.sum((theta_x[nnidx,tidx] - theta_level)**2, 1))
        idx = nnidx[np.where(theta_discrepancy < theta_tol)[0]]
        print("\tAt level {}, len(idx) = {}".format(theta_level,len(idx)),end="")
        if len(idx) == 0:
            print("WARNING: no datapoints are close to the level")
            return [],[],[]
        # Maximize reactive density constrained to the surface
        rflux = Jdn[idx] + Jup[idx] #np.abs(Jdn[idx] + Jup[idx])
        rflux_max_idx = np.where(np.abs(rflux) > frac_of_max*np.max(np.abs(rflux)))[0]
        print(" and len(rflux_max_idx) = {}".format(len(rflux_max_idx)))
        print("rflux = {}".format(rflux))
        #num = min(max_num_states,len(idx))
        #if num < len(idx):
        #    rflux_max_idx = np.argpartition(-rflux,num)[:num]
        #else:
        #    rflux_max_idx = np.arange(len(idx))
        return list(idx[rflux_max_idx]),list(rflux[rflux_max_idx]),theta_x[idx[rflux_max_idx],tidx]
    def maximize_rdens_on_surface(self,comm_bwd,comm_fwd,weight,theta_x,theta_level,theta_tol,max_num_states):
        # Find data closest to a certain level set, and return a set of datapoints
        # First, project reactive current onto the full-state CV space
        #theta_x = theta(data.X).flatten()
        idx = np.where(np.abs(theta_x - theta_level) < theta_tol)[0]
        print("How many datapoints are close to qlevel? {}".format(len(idx)))
        if len(idx) == 0:
            idx = [np.argmin(np.abs(theta_x - theta_level))]
            print("WARNING: no datapoints are close to the level")
        # Maximize reactive density constrained to the surface
        reac_dens = comm_fwd[idx]*comm_bwd[idx]*weight[idx]
        reac_dens *= 1/np.sum(reac_dens)
        num = min(max_num_states,len(idx))
        reac_dens_max_idx = np.argpartition(-reac_dens,num)[:num]
        return idx[reac_dens_max_idx],reac_dens[reac_dens_max_idx],theta_x[idx[reac_dens_max_idx]]
    def plot_median_flux_parametric(self,model,data,ramp,field_x,field_y,units_x=1.0,units_y=1.0,ramp_levels=None,ramp_tol_list=None,fig=None,ax=None,ellipse_flag=False,clip_ab_flag=False):
        Nx,Nt,xdim = data.X.shape
        comm_fwd = self.dam_moments['one']['xb'][0,:,:]
        comm_bwd = self.dam_moments['one']['ax'][0,:,:]
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
        rampflat = ramp.flatten()
        rampflat = rampflat[np.isnan(rampflat)==0]
        if ramp_levels is None or ramp_tol_list is None:
            num_levels = 20 # 17 is the default
            ramp_min,ramp_max = np.nanmin(rampflat),np.nanmax(rampflat)
            rampflat = rampflat[(rampflat>ramp_min)*(rampflat<ramp_max)]
            ramp_edges = np.linspace(ramp_min,ramp_max,num_levels+1) 
            #print(f"ramp_edges = {ramp_edges}")
            ramp_levels = (ramp_edges[:-1] + ramp_edges[1:])/2
            ramp_tol_list = (ramp_edges[1:] - ramp_edges[:-1])/1.0
        # Plot
        if fig is None or ax is None:
            fig,ax = plt.subplots()
        ramp = ramp.reshape((Nx,Nt,1))
        ellipse_list = []
        centers_x = []
        centers_y = []
        ramp_levels_real = []
        # Predetermine the quantiles
        quantiles = np.array([0.05,0.25,0.4,0.5]) #0.9,0.5,0.2])
        quantiles_y_lower = np.nan*np.ones((len(quantiles)-1,len(ramp_levels)))
        quantiles_y_upper = np.nan*np.ones((len(quantiles)-1,len(ramp_levels)))
        quantiles_y_mid = np.nan*np.ones(len(ramp_levels))
        quantiles_x_lower = np.nan*np.ones((len(quantiles)-1,len(ramp_levels)))
        quantiles_x_upper = np.nan*np.ones((len(quantiles)-1,len(ramp_levels)))
        quantiles_x_mid = np.nan*np.ones(len(ramp_levels))
        for ti in range(len(ramp_levels)):
            ramp_tol = ramp_tol_list[ti]
            ridx_ti,rflux_ti,_ = self.maximize_rflux_on_surface(model,data,ramp,comm_bwd,comm_fwd,self.chom,ramp_levels[ti],ramp_tol,None,0.0)
            ridx_ti = np.array(ridx_ti)
            rflux_ti = np.array(rflux_ti)
            pos_idx = np.where(rflux_ti>0)[0]
            ridx_ti = ridx_ti[pos_idx]
            rflux_ti = rflux_ti[pos_idx]
            if len(ridx_ti) > 0:
                ramp_levels_real += [ramp_levels[ti]]
                ridx_ti = np.array(ridx_ti)
                rflux_ti = np.array(rflux_ti)
                f_x = field_x[ridx_ti,tidx]
                f_y = field_y[ridx_ti,tidx]
                # Compute quantiles in y direction
                order = np.argsort(f_y)
                cdf = np.cumsum(rflux_ti[order])
                for qi in range(len(quantiles)-1):
                    quantiles_y_lower[qi,ti] = f_y[order[np.where(cdf >= cdf[-1]*quantiles[qi])[0][0]]]
                    quantiles_y_upper[qi,ti] = f_y[order[np.where(cdf >= cdf[-1]*(1-quantiles[qi]))[0][0]]]
                    if quantiles_y_lower[qi,ti] > quantiles_y_upper[qi,ti]:
                        raise Exception(f"ERROR: at qi={qi}, ti={ti}, the lower and upper y quantiles are {quantiles_y_lower[qi,ti]},{quantiles_y_upper[qi,ti]}")
                quantiles_y_mid[ti] = f_y[order[np.where(cdf >= cdf[-1]*quantiles[-1])[0][0]]]
                # Compute quantiles in x direction
                order = np.argsort(f_x)
                cdf = np.cumsum(rflux_ti[order])
                for qi in range(len(quantiles)-1):
                    quantiles_x_lower[qi,ti] = f_x[order[np.where(cdf >= cdf[-1]*quantiles[qi])[0][0]]]
                    quantiles_x_upper[qi,ti] = f_x[order[np.where(cdf >= cdf[-1]*(1-quantiles[qi]))[0][0]]]
                quantiles_x_mid[ti] = f_x[order[np.where(cdf >= cdf[-1]*quantiles[-1])[0][0]]]
                # Compute median in x direction
                order = np.argsort(f_x)
                cdf = np.cumsum(rflux_ti[order])
                quantiles_x_mid[ti] = f_x[order[np.where(cdf >= cdf[-1]*0.5)[0][0]]]
                # Compute an equivalent ellipse
                flux_sum = np.sum(rflux_ti)
                mean_x = np.sum(f_x*rflux_ti)/flux_sum
                mean_y = np.sum(f_y*rflux_ti)/flux_sum
                centers_x += [mean_x]
                centers_y += [mean_y]
                print(f"mean_x = {mean_x}, mean_y = {mean_y}")
                mean_xx = np.sum(f_x*f_x*rflux_ti)/flux_sum
                mean_xy = np.sum(f_x*f_y*rflux_ti)/flux_sum
                mean_yy = np.sum(f_y*f_y*rflux_ti)/flux_sum
                # Covariance matrix
                C = np.array([[mean_xx-mean_x**2, mean_xy-mean_x*mean_y], [mean_xy-mean_x*mean_y, mean_yy-mean_y**2]])
                # Convert to dimensional form
                D = np.diag([units_x,units_y])
                C_d = D.dot(C).dot(D)
                lam,eig = np.linalg.eigh(C_d)
                order = np.argsort(lam)
                lam = lam[order]
                eig = eig[:,order]
                print(f"lam = {lam}")
                print(f"eig = {eig}")
                if np.min(lam) <= 0:
                    raise Exception(f"ERROR: the covariance matrix C has a nonpositive eigenvalue. lam = {lam}")
                angle = np.arctan2(eig[1,1],eig[0,1])*180/np.pi
                print(f"angle = {angle}")
                for i_sig in range(len(quantiles-1)):
                    # Determine the semi-major and minor axes for this.
                    R = np.sqrt(-2*np.log(2*quantiles[i_sig]))
                    print(f"for quantile {quantiles[i_sig]}, R = {R}")
                    #reds = 1-0.5*quantiles[i_sig]
                    reds = (i_sig+1)/(len(quantiles)+1)
                    ellipse = patches.Ellipse((mean_x*units_x,mean_y*units_y),R*np.sqrt(lam[1]),R*np.sqrt(lam[0]),angle=angle,fc=plt.cm.Reds(reds), fill=True, zorder=i_sig)
                    if ellipse_flag:
                        ax.add_artist(ellipse)
                #ellipse_list += [ellipse]
                # Also draw the ellipse to make sure
                theta = np.linspace(0,2*np.pi,60)
                xx = np.sqrt(lam[1])*np.cos(theta)
                yy = np.sqrt(lam[0])*np.sin(theta)
                ell_edge = np.array([[np.cos(angle*np.pi/180),-np.sin(angle*np.pi/180)],[np.sin(angle*np.pi/180),np.cos(angle*np.pi/180)]]).dot(np.array([xx,yy]))
                #ax.plot(mean_x*units_x+ell_edge[0],mean_y*units_y+ell_edge[1],color='black',linewidth=1)
        #ellipse_collection = PatchCollection(ellipse_list, zorder=1)
        #ax.add_collection(ellipse_collection)
        centers_x = np.array(centers_x)
        centers_y = np.array(centers_y)
        good_tidx_y = np.where(np.any(np.isnan(quantiles_y_lower)+np.isnan(quantiles_y_upper)+np.isnan(quantiles_y_mid), axis=0) == 0)[0]
        good_tidx_x = np.where(np.any(np.isnan(quantiles_x_lower)+np.isnan(quantiles_x_upper)+np.isnan(quantiles_x_mid), axis=0) == 0)[0]
        good_tidx = np.intersect1d(good_tidx_x,good_tidx_y)
        quantiles_y_lower = quantiles_y_lower[:,good_tidx]
        quantiles_y_upper = quantiles_y_upper[:,good_tidx]
        quantiles_y_mid = quantiles_y_mid[good_tidx]
        quantiles_x_lower = quantiles_x_lower[:,good_tidx]
        quantiles_x_upper = quantiles_x_upper[:,good_tidx]
        quantiles_x_mid = quantiles_x_mid[good_tidx]
        if len(np.unique(quantiles_x_mid)) < len(quantiles_x_mid):
            raise Exception(f"ERROR: duplicates in qxm: {quantiles_x_mid}")
        ramp_levels_real = np.array(ramp_levels_real)
        field_y_a,field_y_b = self.out_of_sample_extension(field_y[:,0],data,model.tpt_obs_xst)
        ax.axhline(field_y_a*units_y,color='skyblue')
        ax.axhline(field_y_b*units_y,color='red')
        if ellipse_flag:
            # Plot a center line
            ramp_levels_interp = np.linspace(ramp_levels_real[0],ramp_levels_real[-1],200)
            idx_interp = np.linspace(0,len(ramp_levels_real)-1,len(ramp_levels_real)).astype(int)
            centers_x_interp = scipy.interpolate.interp1d(ramp_levels_real[idx_interp],centers_x[idx_interp],'linear')(ramp_levels_interp)
            centers_y_interp = scipy.interpolate.interp1d(ramp_levels_real[idx_interp],centers_y[idx_interp],'linear')(ramp_levels_interp)
            ax.plot(centers_x_interp*units_x,centers_y_interp*units_y,color='black',linewidth=2, zorder=10)
        else:
            order = np.argsort(quantiles_x_mid)
            quantiles_x_mid = quantiles_x_mid[order]
            quantiles_x_lower = quantiles_x_lower[:,order]
            quantiles_x_upper = quantiles_x_upper[:,order]
            quantiles_y_mid = quantiles_y_mid[order]
            quantiles_y_lower = quantiles_y_lower[:,order]
            quantiles_y_upper = quantiles_y_upper[:,order]
            if clip_ab_flag:
                quantiles_y_mid[-1] = field_y_b
                quantiles_y_upper[:,-1] = field_y_b
                quantiles_y_lower[:,-1] = field_y_b
            qx_interp = np.linspace(quantiles_x_mid.min(),quantiles_x_mid.max(),100)
            # ------ Moving least squares ----
            qymid_interp = helper.moving_least_squares(quantiles_x_mid,quantiles_y_mid,qx_interp,lengthscale=6.0,degree=2)
            # -------- Scipy interp1d ----- 
            #qymid_interp = scipy.interpolate.interp1d(quantiles_x_mid,quantiles_y_mid,'linear')(qx_interp)
            # -------- B-spline ----- 
            #spl = splrep(quantiles_x_mid,quantiles_y_mid)
            #qymid_interp = splev(qx_interp, spl)
            for qi in range(len(quantiles)-1):
                # ---- Moving least squares ---
                qy_dydown_interp = helper.moving_least_squares(quantiles_x_mid,(quantiles_y_mid-quantiles_y_lower[qi]),qx_interp,lengthscale=6.0,degree=2)
                qy_dyup_interp = (helper.moving_least_squares(quantiles_x_mid,(quantiles_y_upper[qi]-quantiles_y_mid),qx_interp,lengthscale=6.0,degree=2))
                # ---- Scipy interp1d ---
                #qy_dydown_interp = np.exp(scipy.interpolate.interp1d(quantiles_x_mid,np.log(quantiles_y_mid-quantiles_y_lower[qi]),'linear')(qx_interp))
                #qy_dyup_interp = np.exp(scipy.interpolate.interp1d(quantiles_x_mid,np.log(quantiles_y_upper[qi]-quantiles_y_mid),'linear')(qx_interp))
                # ----- B-spline ------
                #qy_dydown_interp = np.exp(splev(qx_interp,splrep(quantiles_x_mid,np.log(quantiles_y_mid-quantiles_y_lower[qi]))))
                #qy_dyup_interp = np.exp(splev(qx_interp,splrep(quantiles_x_mid,np.log(quantiles_y_upper[qi]-quantiles_y_mid))))
                #--------- Fill between ------
                lower = qymid_interp-qy_dydown_interp
                upper = qymid_interp+qy_dyup_interp
                if clip_ab_flag:
                    lower = np.maximum(lower, min(field_y_a,field_y_b))
                    upper = np.minimum(upper, max(field_y_a,field_y_b))
                ax.fill_between(qx_interp*units_x,lower*units_y,upper*units_y,color=plt.cm.Reds((qi+1)/len(quantiles)),alpha=1.0,zorder=qi)
            mid = qymid_interp
            if clip_ab_flag:
                mid = np.maximum(mid, min(field_y_a,field_y_b))
                mid = np.minimum(mid, max(field_y_a,field_y_b))
            ax.plot(qx_interp*units_x,mid*units_y,color='black')
            #ax.scatter(quantiles_x_mid*units_x,quantiles_y_mid*units_y,marker='o',color='black',zorder=2*len(quantiles))
        return fig,ax
    def plot_median_flux_and_lap_signed(self,model,data,ramp,field,field_fun=None,ramp_units=1.0,field_units=1.0,ramp_levels=None,ramp_tol_list=None,fig=None,ax=None,laptime_flag=False,field_twin=None,twin_name=None):
        # Just scalars. 
        Nx,Nt,xdim = data.X.shape
        comm_fwd = self.dam_moments['one']['xb'][0,:,:]
        comm_bwd = self.dam_moments['one']['ax'][0,:,:]
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
        funlib = model.observable_function_library()
        rampflat = ramp.flatten()
        rampflat = rampflat[np.isnan(rampflat)==0]
        if ramp_levels is None or ramp_tol_list is None:
            num_levels = 20 # 17 is the default
            ramp_min,ramp_max = np.nanmin(rampflat),np.nanmax(rampflat)
            rampflat = rampflat[(rampflat>ramp_min)*(rampflat<ramp_max)]
            ramp_edges = np.linspace(ramp_min,ramp_max,num_levels+1) 
            #print(f"ramp_edges = {ramp_edges}")
            ramp_levels = (ramp_edges[:-1] + ramp_edges[1:])/2
            ramp_tol_list = (ramp_edges[1:] - ramp_edges[:-1])/1.0
        # Plot
        if fig is None or ax is None:
            fig,ax = plt.subplots()
        ramp = ramp.reshape((Nx,Nt,1))
        bin_width = (np.nanmax(field)-np.nanmin(field))/30
        print(f"bin_width = {bin_width}")
        twin_vals = []
        ramp_levels_real = []
        for ti in range(len(ramp_levels)):
            ramp_tol = ramp_tol_list[ti]
            ridx_ti,rflux_ti,_ = self.maximize_rflux_on_surface(model,data,ramp,comm_bwd,comm_fwd,self.chom,ramp_levels[ti],ramp_tol,None,0.0)
            ridx_ti = np.array(ridx_ti)
            rflux_ti = np.array(rflux_ti)
            #pos_idx = np.where(rflux_ti>0)[0]
            #ridx_ti = ridx_ti[pos_idx]
            #rflux_ti = rflux_ti[pos_idx]
            if len(ridx_ti) > 0:
                ramp_levels_real += [ramp_levels[ti]]
                ridx_ti = np.array(ridx_ti)
                rflux_ti = np.array(rflux_ti)
                f_ti = field[ridx_ti,tidx]
                order = np.argsort(f_ti)
                f_ti = f_ti[order]
                rflux_ti = rflux_ti[order]
                sig_idx = np.where(np.abs(rflux_ti) > 0.0*np.max(np.abs(rflux_ti)))[0]
                f_ti = f_ti[sig_idx]
                if field_twin is not None: 
                    g_ti = field_twin[ridx_ti[sig_idx],tidx]
                    twin_vals += [np.sum(g_ti*rflux_ti)/np.sum(rflux_ti)]
                rflux_ti = rflux_ti[sig_idx]
                num_bins = int(round((np.max(f_ti) - np.min(f_ti))/bin_width))
                hist,bin_edges = np.histogram(f_ti,weights=rflux_ti,density=False,bins=num_bins) #min(10,len(sig_idx)))
                bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
                hist *= 0.8*(ramp_levels[1]-ramp_levels[0])*ramp_units/np.max(np.abs(hist))
                # Truncate bin centers to nontrivial fluxes
                sig_bin_idx = np.where(np.abs(hist) > 0.1*np.max(np.abs(hist)))[0]
                bin_centers = bin_centers[sig_bin_idx]
                hist = hist[sig_bin_idx]
                ax.plot(ramp_levels[ti]*ramp_units*np.ones(len(bin_centers)), bin_centers*field_units,color='black',linewidth=1)
                ax.barh(bin_centers*field_units,hist,(bin_edges[1]-bin_edges[0])*field_units,left=ramp_levels[ti]*ramp_units,color='gray',alpha=0.8)
                #ax.plot(ramp_levels[ti]*ramp_units+hist,bin_centers*field_units,color='black',linewidth=1)
                #ax.fill_betweenx(bin_centers*field_units,ramp_levels[ti]*ramp_units,ramp_levels[ti]*ramp_units+hist,color='gray',alpha=0.5)
                #print(f"len(unique(f_ti)) = {len(np.unique(f_ti))}, len(unique(rflux_ti)) = {len(np.unique(rflux_ti))},") #\nf_ti = {f_ti},\nrflux_ti = {rflux_ti}")
                #print(f"len(sig_idx) = {len(sig_idx)}, len(bin_centers) = {len(bin_centers)}")
                #print(f"bin_centers*field_units = {bin_centers*field_units},\nhist*ramp_units = {hist*ramp_units}")
        # ----------------- Make a secondary axis -----------
        if field_twin is not None:
            ramp_levels_real = np.array(ramp_levels_real)*ramp_units
            twin_vals = np.array(twin_vals)
            reg = linear_model.LinearRegression()
            print(f"twin_vals = {twin_vals}, reg = {reg}")
            reg.fit(ramp_levels_real.reshape(-1,1),twin_vals)
            def field2twin(r):
                return reg.intercept_ + reg.coef_[0]*r
            def twin2field(t):
                return (t - reg.intercept_)/reg.coef_[0]
            secax = ax.secondary_xaxis('top', functions=(field2twin,twin2field))
            secax.set_xlabel(twin_name,fontdict=ffont)
            secax.tick_params(labelsize=10)
        # Least action path 
        xlap = load(join(self.physical_param_folder,"xmin_dirn1.npy"))
        tlap = load(join(self.physical_param_folder,"tmin_dirn1.npy"))
        adist_lap = model.adist(xlap)
        bdist_lap = model.bdist(xlap)
        tlap_idx0 = np.where((adist_lap>0)*(bdist_lap>0))[0][0]
        if np.min(bdist_lap) <= 0:
            tlap_idx1 = np.where(bdist_lap<=0)[0][0]
        else:
            tlap_idx1 = np.where((adist_lap>0)*(bdist_lap>0))[0][-1]
        print(f"tlap_idx0 = {tlap_idx0}, tlap_idx1 = {tlap_idx1}")
        ramp = ramp.reshape((Nx,Nt,1))
        tlap = tlap[tlap_idx0:tlap_idx1+1]
        xlap = xlap[tlap_idx0:tlap_idx1+1]
        tlap -= tlap[-1]
        # Subsample
        xlap = xlap[np.linspace(0,len(xlap)-1,10).astype(int)]
        tlap = tlap[np.linspace(0,len(tlap)-1,10).astype(int)]
        # Interpolate the field onto the least-action path
        if field_fun is None:
            f_lap = self.out_of_sample_extension(field[:,0],data,xlap)
        else:
            f_lap = field_fun(xlap)
        ramp_lap = self.out_of_sample_extension(ramp[:,0,0],data,xlap)
        tlap_interp = np.linspace(tlap[0],tlap[-1],100)
        ramp_lap_interp = scipy.interpolate.interp1d(tlap,ramp_lap,kind='cubic')(tlap_interp)
        idx = np.where(ramp_lap_interp > ramp_levels[0])[0]
        f_lap_interp = scipy.interpolate.interp1d(tlap,f_lap,kind='cubic')(tlap_interp)
        ax.plot(ramp_lap_interp[idx]*ramp_units,f_lap_interp[idx]*field_units,color='black',linestyle='--')
        # Smooth the least-action path
        if laptime_flag:
            ax.plot(tlap,f_lap*field_units,color='cyan',linestyle='-')
        if field_fun is None:
            field_a,field_b = self.out_of_sample_extension(field[:,0],data,model.tpt_obs_xst)
        else:
            field_a,field_b = field_fun(model.tpt_obs_xst)
        ax.axhline(y=field_a*field_units,color='skyblue',linewidth=3)
        ax.axhline(y=field_b*field_units,color='red',linewidth=3)
        #fig.savefig(join(self.savefolder,"lap_vs_tpt_ab_profiles_U_vs_qp"),bbox_inches="tight",pad_inches=0.2)
        #plt.close(fig)
        return fig,ax
    def plot_median_flux_and_lap(self,model,data,ramp,field,field_fun=None,ramp_units=1.0,field_units=1.0,ramp_levels=None,ramp_tol_list=None,fig=None,ax=None,laptime_flag=False):
        # Just scalars. 
        Nx,Nt,xdim = data.X.shape
        comm_fwd = self.dam_moments['one']['xb'][0,:,:]
        comm_bwd = self.dam_moments['one']['ax'][0,:,:]
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
        quantiles = 0.01*np.array([5.0,25.0,40.0,50.0,60.0,75.0,95.0])
        funlib = model.observable_function_library()
        rampflat = ramp.flatten()
        rampflat = rampflat[np.isnan(rampflat)==0]
        if ramp_levels is None or ramp_tol_list is None:
            num_levels = 20 # 17 is the default
            ramp_min,ramp_max = np.nanmin(rampflat),np.nanmax(rampflat)
            rampflat = rampflat[(rampflat>ramp_min)*(rampflat<ramp_max)]
            ramp_edges = np.linspace(ramp_min,ramp_max,num_levels+1) 
            #print(f"ramp_edges = {ramp_edges}")
            ramp_levels = (ramp_edges[:-1] + ramp_edges[1:])/2
            ramp_tol_list = (ramp_edges[1:] - ramp_edges[:-1])/1.0
        rflux = []
        rflux_idx = []
        field_quants = []
        ramp_levels_real = []
        ramp = ramp.reshape((Nx,Nt,1))
        for ti in range(len(ramp_levels)):
            ramp_tol = ramp_tol_list[ti]
            ridx_ti,rflux_ti,_ = self.maximize_rflux_on_surface(model,data,ramp,comm_bwd,comm_fwd,self.chom,ramp_levels[ti],ramp_tol,None,0.0)
            ridx_ti = np.array(ridx_ti)
            rflux_ti = np.array(rflux_ti)
            # *** Restrict to positive-flux areas if you want to avoid weirdness on boundaries of A and B *** 
            #pos_idx = np.where(rflux_ti>0)[0]
            #ridx_ti = ridx_ti[pos_idx]
            #rflux_ti = rflux_ti[pos_idx]
            if len(ridx_ti) > 0:
                ridx_ti = np.array(ridx_ti)
                rflux_ti = np.array(rflux_ti)
                rflux += [rflux_ti]
                rflux_idx += [ridx_ti]
                ramp_levels_real += [ramp_levels[ti]]
                f_ti = field[ridx_ti,tidx]
                fq_ti = np.zeros(len(quantiles))
                order = np.argsort(f_ti)
                cdf = np.cumsum(rflux_ti[order])
                for qi in range(len(quantiles)):
                    quant_idx = np.where(cdf/cdf[-1] > quantiles[qi])[0][0]
                    fq_ti[qi] = f_ti[order[quant_idx]]
                field_quants += [fq_ti]
        ramp_levels_real = np.array(ramp_levels_real)
        print("len(ramp_levels_real) = {}".format(len(ramp_levels_real)))
        print(f"ramp_levels_real = {ramp_levels_real}")
        field_quants = np.array(field_quants)
        # Least action path 
        xlap = load(join(self.physical_param_folder,"xmin_dirn1.npy"))
        tlap = load(join(self.physical_param_folder,"tmin_dirn1.npy"))
        adist_lap = model.adist(xlap)
        bdist_lap = model.bdist(xlap)
        tlap_idx0 = np.where((adist_lap>0)*(bdist_lap>0))[0][0]
        if np.min(bdist_lap) <= 0:
            tlap_idx1 = np.where(bdist_lap<=0)[0][0]
        else:
            tlap_idx1 = np.where((adist_lap>0)*(bdist_lap>0))[0][-1]
        print(f"tlap_idx0 = {tlap_idx0}, tlap_idx1 = {tlap_idx1}")
        tlap = tlap[tlap_idx0:tlap_idx1+1]
        xlap = xlap[tlap_idx0:tlap_idx1+1]
        tlap -= tlap[-1]
        # Subsample
        xlap = xlap[np.linspace(0,len(xlap)-1,10).astype(int)]
        tlap = tlap[np.linspace(0,len(tlap)-1,10).astype(int)]
        # Interpolate the field onto the least-action path
        if field_fun is None:
            f_lap = self.out_of_sample_extension(field[:,0],data,xlap)
        else:
            f_lap = field_fun(xlap)
        ramp_lap = self.out_of_sample_extension(ramp[:,0,0],data,xlap)
        # Interpolate the least action path to smooth out
        tlap_interp = np.linspace(tlap[0],tlap[-1],100)
        ramp_lap_interp = scipy.interpolate.interp1d(tlap,ramp_lap,kind='cubic')(tlap_interp)
        f_lap_interp = scipy.interpolate.interp1d(tlap,f_lap,kind='cubic')(tlap_interp)
        # Plot
        if fig is None or ax is None:
            fig,ax = plt.subplots()
        ax.plot(ramp_lap_interp*ramp_units,f_lap_interp*field_units,color='black',linestyle='--')
        if laptime_flag:
            ax.plot(tlap,f_lap*field_units,color='cyan',linestyle='-')
        levels_interp = np.linspace(ramp_levels_real[0],ramp_levels_real[-1],200)
        field_quant_interp = np.zeros((len(quantiles),len(levels_interp)))
        for qi in range(len(quantiles)):
            field_quant_interp[qi] = scipy.interpolate.interp1d(ramp_levels_real,field_quants[:,qi],kind='cubic')(levels_interp)
        med_qi = len(quantiles)//2
        for qi in range(med_qi): # Assume an odd number with the middle is 50%
            ax.fill_between(levels_interp,field_quant_interp[qi]*field_units,field_quant_interp[len(quantiles)-1-qi]*field_units,color=plt.cm.Reds((qi+1)/len(quantiles)),alpha=1.0,zorder=qi)
        ax.plot(levels_interp,field_quant_interp[med_qi]*field_units,color='black',zorder=med_qi)
        ax.scatter(ramp_levels_real,field_quants[:,med_qi]*field_units,color='black',marker='o',zorder=med_qi)
        print(f"black dots: {field_quants[:,med_qi]}")
        if field_fun is None:
            field_a,field_b = self.out_of_sample_extension(field[:,0],data,model.tpt_obs_xst)
        else:
            field_a,field_b = field_fun(model.tpt_obs_xst)
        ax.axhline(y=field_a*field_units,color='skyblue',linewidth=3)
        ax.axhline(y=field_b*field_units,color='red',linewidth=3)
        #fig.savefig(join(self.savefolder,"lap_vs_tpt_ab_profiles_U_vs_qp"),bbox_inches="tight",pad_inches=0.2)
        #plt.close(fig)
        return fig,ax
    def plot_transition_states_ensttend(self,model,data):
        # Plot the enstrophy tendency according to (1) deterministic model, (2) steady-state current, (3) reactive current
        Nx,Nt,xdim = data.X.shape
        n = model.q['Nz']-1
        comm_fwd = self.dam_moments['one']['xb'][0,:,:]
        comm_bwd = self.dam_moments['one']['ax'][0,:,:]
        ramp = comm_fwd.reshape((Nx,Nt,1))
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
        qp_levels = np.array([0.00,0.2,0.5,0.8,1.0])
        colors = np.array([plt.cm.coolwarm(qpl) for qpl in qp_levels])
        colors[np.abs(qp_levels - 0.5) < 0.01] = matplotlib.colors.to_rgba('orange')
        qp_tol_list = 0.1*np.ones(len(qp_levels))
        labels = [r"$q^+=%.2f$"%(0.5*(
            min(1, max(0, qp_levels[i]-qp_tol_list[i])) + 
            min(1, max(0, qp_levels[i]+qp_tol_list[i])))) 
            for i in range(len(qp_levels))]
        rflux = []
        rflux_idx = []
        for qi in range(len(qp_levels)):
            qp_tol = qp_tol_list[qi]
            ridx_qi,rflux_qi,_ = self.maximize_rflux_on_surface(model,data,ramp,comm_bwd,comm_fwd,self.chom,qp_levels[qi],qp_tol,None,0.0)
            rflux += [rflux_qi]
            rflux_idx += [ridx_qi]
        fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(18,12))
        model.plot_state_distribution(data.X[:,tidx],rflux,rflux_idx,qp_levels,r"$q^+$",key="enstproj",colors=colors,labels=labels,fig=fig,ax=ax[0,0])
        ax[0,0].set_title("Eddy enstrophy")
        # Now project the current operator onto each level of enstrophy
        funlib = model.observable_function_library()
        enstproj = funlib["enstproj"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,n))
        Lab,tidx_ab = self.tendency_during_transition(model,data,enstproj,comm_bwd,comm_fwd)
        L,tidx = self.tendency_during_transition(model,data,enstproj,np.ones_like(comm_bwd),np.ones_like(comm_fwd))
        Jab_up,Jab_dn = self.project_current_new(model,data,enstproj,comm_bwd,comm_fwd)
        J_up,J_dn = self.project_current_new(model,data,enstproj,np.ones_like(comm_bwd),np.ones_like(comm_fwd))
        z = model.q['z_d'][1:-1]/1000
        for qi in range(len(qp_levels)):
            J = np.sum(((J_up[rflux_idx[qi]] + J_dn[rflux_idx[qi]]).T/2 * rflux[qi]).T, axis=0) / np.nansum(rflux_qi)
            Jab = np.sum(((Jab_up[rflux_idx[qi]] + Jab_dn[rflux_idx[qi]]).T/2 * rflux[qi]).T, axis=0) / np.nansum(rflux_qi)
            ax[0,1].plot(J,z,color=colors[qi])
            ax[0,1].set_xlabel(r"$J\cdot\nabla(\frac{1}{2}\overline{q'^2})$")
            ax[0,2].plot(Jab,z,color=colors[qi])
            ax[0,2].set_xlabel(r"$J_{AB}\cdot\nabla(\frac{1}{2}\overline{q'^2})$")
            # Now tendencies
            nnidx = np.where(np.any(np.isnan(L[rflux_idx[qi]]),axis=1)==0)[0]
            print(f"nnidx.shape = {nnidx.shape}")
            L_qi = np.sum((L[rflux_idx[qi]][nnidx].T * self.chom[rflux_idx[qi]][nnidx]).T, axis=0)/np.sum(self.chom[rflux_idx[qi]][nnidx])
            Lab_qi = np.nansum((Lab[rflux_idx[qi]][nnidx].T * self.chom[rflux_idx[qi]][nnidx]).T, axis=0)/np.sum(self.chom[rflux_idx[qi]][nnidx])
            ax[1,1].plot(L_qi*funlib["enstproj"]["units"],z,color=colors[qi])
            ax[1,1].set_xlabel(r"$\partial_t(\frac{1}{2}\overline{q'^2})$ [%s day$^{-1}$]"%(funlib["enstproj"]["unit_symbol"]))
            ax[1,1].set_title("Steady-state")
            ax[1,2].plot(Lab_qi*funlib["enstproj"]["units"],z,color=colors[qi])
            ax[1,2].set_xlabel(r"$\partial_t(\frac{1}{2}\overline{q'^2})$ [%s day$^{-1}$]"%(funlib["enstproj"]["unit_symbol"]))
            ax[1,2].set_title(r"$x\to B$")
            # Now plot the deterministic tendency
            ensttend = funlib["ensttend"]["fun"](data.X[rflux_idx[qi]][nnidx][:,tidx])
            L_deterministic = np.sum((ensttend.T * self.chom[rflux_idx[qi]][nnidx]).T, axis=0).T/np.sum(self.chom[rflux_idx[qi]][nnidx])
            ax[1,0].plot(L_deterministic*funlib["ensttend"]["units"]*model.q["time"],z,color=colors[qi])
            ax[1,0].set_xlabel(r"$\partial_t(\frac{1}{2}\overline{q'^2})$ [%s day$^{-1}$]"%(funlib["enstproj"]["unit_symbol"]))
            ax[1,0].set_title("Deterministic")
        fig.savefig(join(self.savefolder,f"trans_state_profile_enstproj_ABnormal"))
        return
    def plot_transition_states_new(self,model,data):
        # All new version. One straightforward function. Plot max-flux path, and also plot profiles. 
        composite_flag = True 
        plot_profile_flag = True 
        parametric_flag = True 
        signed_flag = True 
        unsigned_flag = True 
        # ------------------------------ 1. For several committor levels, plot the profile of zonal wind and heat flux. ----------------------------------
        Nx,Nt,xdim = data.X.shape
        comm_fwd = self.dam_moments['one']['xb'][0,:,:]
        comm_bwd = self.dam_moments['one']['ax'][0,:,:]
        # --------------------
        # Test on a and b levels
        funlib = model.observable_function_library()
        zi = model.q['zi']
        Uzi_a,Uzi_b = funlib["U"]["fun"](model.tpt_obs_xst)[:,zi]
        print("Uzi_b*units = {}".format(Uzi_b*funlib["U"]["units"]))
        print("(Uzi_b + radius_b)*units = {}".format((Uzi_b+model.radius_b)*funlib["U"]["units"]))
        #---------------
        # Quick test: do any points have bdist=0 and lead time > 0?
        funlib = model.observable_function_library()
        eps = 0.01
        tb = self.dam_moments['one']['xb'][1,:,:]
        tb = tb*(comm_fwd > eps)/(comm_fwd + 1.0*(comm_fwd <= eps))
        tb[comm_fwd <= eps] == np.nan
        bdist = model.bdist(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
        U = funlib["Uref"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
        bbndy = funlib["Uref"]["fun"](model.tpt_obs_xst)[1] + model.radius_b
        numbad = np.nansum((bdist==0)*(tb>0))
        nb = np.sum(bdist<=0)
        nsmallu = np.sum(U < bbndy)
        print("nb = {}, nsmallu = {}, numbad = {}".format(nb,nsmallu,numbad))
        # ---------------------
        ramp = comm_fwd.reshape((Nx,Nt,1))
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
        qp_levels = np.array([0.00,0.2,0.5,0.8,1.0])
        #colors = np.array([plt.cm.coolwarm(qp_levels[0]),'orange',plt.cm.coolwarm(qp_levels[2])])
        colors = np.array([plt.cm.coolwarm(qpl) for qpl in qp_levels])
        colors[np.abs(qp_levels - 0.5) < 0.01] = matplotlib.colors.to_rgba('orange')
        qp_tol_list = 0.1*np.ones(len(qp_levels))
        labels = [r"$q^+=%.2f$"%(0.5*(
            min(1, max(0, qp_levels[i]-qp_tol_list[i])) + 
            min(1, max(0, qp_levels[i]+qp_tol_list[i]))))
            for i in range(len(qp_levels))]
        if plot_profile_flag:
            #prof_key_list = ["U","vT","dqdy","q2"]
            prof_key_list = ["U","dissproj","dqdyproj","enstproj","vqproj","ensttend"]
            rflux = []
            rflux_idx = []
            for qi in range(len(qp_levels)):
                qp_tol = qp_tol_list[qi]
                ridx_qi,rflux_qi,_ = self.maximize_rflux_on_surface(model,data,ramp,comm_bwd,comm_fwd,self.chom,qp_levels[qi],qp_tol,None,0.0)
                rflux += [rflux_qi]
                rflux_idx += [ridx_qi]
            # Signed 
            for ki in range(len(prof_key_list)):
                prof_key = prof_key_list[ki]
                fig,ax = model.plot_state_distribution_signed(data.X[:,tidx],rflux,rflux_idx,qp_levels,r"$q^+$",key=prof_key,colors=colors,labels=labels)
                fig.savefig(join(self.savefolder,"trans_state_profile_signed_{}".format(prof_key)),bbox_inches="tight",pad_inches=0.2)
                plt.close(fig)
            # Unsigned 
            for ki in range(len(prof_key_list)):
                prof_key = prof_key_list[ki]
                fig,ax = model.plot_state_distribution(data.X[:,tidx],rflux,rflux_idx,qp_levels,r"$q^+$",key=prof_key,colors=colors,labels=labels)
                fig.savefig(join(self.savefolder,"trans_state_profile_{}".format(prof_key)),bbox_inches="tight",pad_inches=0.2)
                plt.close(fig)
        # ---------------------- 2. Plot scalar path distributions ----------------
        # Now plot a few ramps
        eps = 1e-10
        tb = self.dam_moments['one']['xb'][1,:,:]
        tb = tb*(comm_fwd > eps)/(comm_fwd + 1.0*(comm_fwd <= eps))
        tb[comm_fwd <= eps] == np.nan
        tbramp = -tb.reshape((Nx,Nt))
        tbramp_min = -120.0 #np.nanmin(-tb)
        tbramp_max = np.nanmax(-tb)
        tbramp_levels = np.linspace(tbramp_min,tbramp_max,15)
        tbramp_tol_list = np.zeros(len(tbramp_levels))
        tbramp_tol_list[:-1] = (tbramp_levels[1:] - tbramp_levels[:-1])/2
        tbramp_tol_list[-1] = (tbramp_levels[-1] - tbramp_levels[-2])/10
        qpramp = comm_fwd.reshape((Nx,Nt))
        qpramp_min = 0.1
        qpramp_max = 1.0
        qpramp_levels = np.linspace(qpramp_min,qpramp_max,15)
        qpramp_tol_list = np.zeros(len(qpramp_levels))
        qpramp_tol_list[:-1] = (qpramp_levels[1:] - qpramp_levels[:-1])/2
        qpramp_tol_list[-1] = (qpramp_levels[-1] - qpramp_levels[-2])/2 #15
        #ramp_levels[-1] = (ramp_levels[-1] + ramp_levels[-2])/2
        # -------- Parametric ------------
        print(f"------------------------ Beginning parametric ----------------------")
        if parametric_flag:
            # ___ vs. lead time (parametric)
            ellipse_flag = False
            nqpramp = 30
            qpramp_levels_parametric = np.linspace(qpramp_min,qpramp_max,nqpramp)
            dqpramp = (qpramp_max-qpramp_min)/(nqpramp-1)
            #qpramp_levels_parametric = np.concatenate((qpramp_levels_parametric,np.linspace(qpramp_levels_parametric[-2],qpramp_levels_parametric[-1],4)[1:-1]))
            #qpramp_levels_parametric = np.sort(qpramp_levels_parametric)
            qpramp_tol_list_parametric = np.zeros(len(qpramp_levels_parametric))
            qpramp_tol_list_parametric[1:-1] = dqpramp/2
            qpramp_tol_list_parametric[:1] = dqpramp/15
            qpramp_tol_list_parametric[-1:] = dqpramp/15
            field_x = tbramp
            # U vs. lead time 
            field_y = funlib["Uref"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            fig,ax = self.plot_median_flux_parametric(model,data,qpramp,field_x,field_y,units_x=1.0,units_y=funlib["Uref"]["units"],ramp_levels=qpramp_levels_parametric,ramp_tol_list=qpramp_tol_list_parametric,ellipse_flag=ellipse_flag,clip_ab_flag=True)
            ax.set_xlabel(r"$-\eta_B^+\mathrm{ [days]}$",fontdict=ffont)
            #ax.set_xlim([-90,0])
            ax.set_ylabel("%s [%s]"%(funlib["Uref"]["name"],funlib["Uref"]["unit_symbol"]),fontdict=ffont)
            fig.savefig(join(self.savefolder,"lap_vs_tpt_parametric_Uref_vs_tb_nlev{}_ell{}".format(len(qpramp_levels_parametric),int(ellipse_flag))),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # vTint vs. lead time
            field_y = funlib["vTintref"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            fig,ax = self.plot_median_flux_parametric(model,data,qpramp,field_x,field_y,units_x=1.0,units_y=funlib["vTintref"]["units"],ramp_levels=qpramp_levels_parametric,ramp_tol_list=qpramp_tol_list_parametric,ellipse_flag=ellipse_flag,clip_ab_flag=False)
            ax.set_xlabel(r"$-\eta_B^+\mathrm{ [days]}$",fontdict=ffont)
            #ax.set_xlim([-80,0])
            ax.set_ylabel("%s [%s]"%(funlib["vTintref"]["name"],funlib["vTintref"]["unit_symbol"]),fontdict=ffont)
            fig.savefig(join(self.savefolder,"lap_vs_tpt_parametric_vTintref_vs_tb_nlev{}_ell{}".format(len(qpramp_levels_parametric),int(ellipse_flag))),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
        # ------ Signed: Plot committor and lead time against each other ----------
        print(f"-------------------- Beginning signed ------------------")
        if signed_flag:
            # Multiple things
            fig,ax = plt.subplots(ncols=2,figsize=(16,6))
            # lead time vs. committor
            _,_ = self.plot_median_flux_and_lap_signed(model,data,qpramp,-tb,field_fun=None,field_units=1.0,ramp_levels=qpramp_levels,ramp_tol_list=qpramp_tol_list,fig=fig,ax=ax[0])
            ax[0].set_ylabel(r"$-\eta_B^+$ [days]",fontdict=ffont)
            ax[0].set_xlabel(r"$q^+_B$",fontdict=ffont)
            # committor vs. lead time
            _,_ = self.plot_median_flux_and_lap_signed(model,data,tbramp,qpramp,field_fun=None,field_units=1.0,ramp_levels=tbramp_levels,ramp_tol_list=tbramp_tol_list,fig=fig,ax=ax[1],laptime_flag=True)
            ax[1].set_ylabel(r"$q^+_B$",fontdict=ffont)
            ax[1].set_xlabel(r"$-\eta^+_B$",fontdict=ffont)
            for i in range(2): ax[i].tick_params(axis='both',labelsize=15)
            fig.savefig(join(self.savefolder,"lap_vs_tpt_qptb_signed"),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # Plot vs. lead time
            fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(16,12),sharex='col',sharey='row')
            # Uref vs. (committor, lead time)
            field = funlib["Uref"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            _,_ = self.plot_median_flux_and_lap_signed(model,data,qpramp,field,field_fun=funlib["Uref"]["fun"],field_units=funlib["Uref"]["units"],ramp_levels=qpramp_levels,ramp_tol_list=qpramp_tol_list,fig=fig,ax=ax[0,0])
            #_,_  = self.plot_median_flux_and_lap_signed(model,data,tbramp,field,field_fun=funlib["Uref"]["fun"],field_units=funlib["Uref"]["units"],ramp_levels=tbramp_levels,ramp_tol_list=tbramp_tol_list,fig=fig,ax=ax[0,1],laptime_flag=True)
            ax[0,0].set_ylabel("%s [%s]"%(funlib["Uref"]["name"],funlib["Uref"]["unit_symbol"]),fontdict=ffont)
            # vTntref vs. (committor, lead time)
            field = funlib["vTintref"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            _,_ = self.plot_median_flux_and_lap_signed(model,data,qpramp,field,field_fun=funlib["vTintref"]["fun"],field_units=funlib["vTintref"]["units"],ramp_levels=qpramp_levels,ramp_tol_list=qpramp_tol_list,fig=fig,ax=ax[1,0]) #,field_twin=tbramp, twin_name=r"$-\eta_B^+$ [days]")
            #_,_ = self.plot_median_flux_and_lap_signed(model,data,tbramp,field,field_fun=funlib["vTintref"]["fun"],field_units=funlib["vTintref"]["units"],ramp_levels=tbramp_levels,ramp_tol_list=tbramp_tol_list,fig=fig,ax=ax[1,1],laptime_flag=True)
            ax[1,0].set_ylabel("%s [%s]"%(funlib["vTintref"]["name"],funlib["vTintref"]["unit_symbol"]),fontdict=ffont)
            ax[1,0].set_xlabel(r"$q^+_B$",fontdict=ffont)
            ax[1,1].set_xlabel(r"$-\eta^+_B$ [days]",fontdict=ffont)
            for i in range(2):
                for j in range(2):
                    ax[i,j].tick_params(axis='both',labelsize=15)
            fig.savefig(join(self.savefolder,"lap_vs_tpt_all_vs_qptb_signed"),bbox_inches='tight',pad_inches=0.2)
            plt.close(fig)
        # ------ Unsigned: Plot committor and lead time against each other ----------
        if unsigned_flag:
            print(f"-------------------- Beginning unsigned ------------------")
            fig,ax = plt.subplots(ncols=2,figsize=(16,6))
            # lead time vs. committor
            _,_ = self.plot_median_flux_and_lap(model,data,qpramp,-tb,field_fun=None,field_units=1.0,ramp_levels=qpramp_levels,ramp_tol_list=qpramp_tol_list,fig=fig,ax=ax[0])
            ax[0].set_ylabel(r"$-\eta_B^+$ [days]",fontdict=ffont)
            ax[0].set_xlabel(r"$q^+_B$",fontdict=ffont)
            # committor vs. lead time
            _,_ = self.plot_median_flux_and_lap(model,data,tbramp,qpramp,field_fun=None,field_units=1.0,ramp_levels=tbramp_levels,ramp_tol_list=tbramp_tol_list,fig=fig,ax=ax[1],laptime_flag=True)
            ax[1].set_ylabel(r"$q^+_B$",fontdict=ffont)
            ax[1].set_xlabel(r"$-\eta^+_B$",fontdict=ffont)
            for i in range(2): ax[i].tick_params(axis='both',labelsize=15)
            fig.savefig(join(self.savefolder,"lap_vs_tpt_qptb"),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # Plot vs. lead time
            fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(16,12),sharex='col',sharey='row')
            # Uref vs. (committor, lead time)
            field = funlib["Uref"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            _,_ = self.plot_median_flux_and_lap(model,data,qpramp,field,field_fun=funlib["Uref"]["fun"],field_units=funlib["Uref"]["units"],ramp_levels=qpramp_levels,ramp_tol_list=qpramp_tol_list,fig=fig,ax=ax[0,0])
            _,_  = self.plot_median_flux_and_lap(model,data,tbramp,field,field_fun=funlib["Uref"]["fun"],field_units=funlib["Uref"]["units"],ramp_levels=tbramp_levels,ramp_tol_list=tbramp_tol_list,fig=fig,ax=ax[0,1],laptime_flag=True)
            ax[0,0].set_ylabel("%s [%s]"%(funlib["Uref"]["name"],funlib["Uref"]["unit_symbol"]),fontdict=ffont)
            # vTntref vs. (committor, lead time)
            field = funlib["vTintref"]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            _,_ = self.plot_median_flux_and_lap(model,data,qpramp,field,field_fun=funlib["vTintref"]["fun"],field_units=funlib["vTintref"]["units"],ramp_levels=qpramp_levels,ramp_tol_list=qpramp_tol_list,fig=fig,ax=ax[1,0])
            _,_ = self.plot_median_flux_and_lap(model,data,tbramp,field,field_fun=funlib["vTintref"]["fun"],field_units=funlib["vTintref"]["units"],ramp_levels=tbramp_levels,ramp_tol_list=tbramp_tol_list,fig=fig,ax=ax[1,1],laptime_flag=True)
            ax[1,0].set_ylabel("%s [%s]"%(funlib["vTintref"]["name"],funlib["vTintref"]["unit_symbol"]),fontdict=ffont)
            ax[1,0].set_xlabel(r"$q^+_B$",fontdict=ffont)
            ax[1,1].set_xlabel(r"$-\eta^+_B$ [days]",fontdict=ffont)
            for i in range(2):
                for j in range(2):
                    ax[i,j].tick_params(axis='both',labelsize=15)
            fig.savefig(join(self.savefolder,"lap_vs_tpt_all_vs_qptb"),bbox_inches='tight',pad_inches=0.2)
            plt.close(fig)
        return
    def plot_transition_states_all(self,model,data,collect_flag=True):
        for dirn in ['ab']: #,'ba']:
            frac_of_max = 0.0
            # First plot the profiles with a small number of levels
            num_levels = 3
            num_per_level = 5
            if collect_flag: 
                _ = self.collect_transition_states(model,data,'committor',dirn,num_per_level,num_levels,frac_of_max=frac_of_max,tolerance=0.05,ramp_bounds=[0.1,0.9])
                _ = self.collect_transition_states(model,data,'leadtime',dirn,num_per_level,num_levels,frac_of_max=frac_of_max,tolerance=5.0,ramp_bounds=[0.0,1.0])
            for func_key in ["U","vT"]:
                self.plot_transition_states(model,data,'committor',dirn,num_per_level,num_levels,frac_of_max=frac_of_max,func_key=func_key)
                self.plot_transition_states(model,data,'leadtime',dirn,num_per_level,num_levels,frac_of_max=frac_of_max,func_key=func_key)

            # Next plot the evolution
            num_levels = 15
            num_per_level = 5
            Nx,Nt,xdim = data.X.shape
            funlib = model.observable_function_library()
            ramp_projection = None
            #ramp_projection = np.array([funlib["magref"]["fun"](data.X.reshape((Nx*Nt,xdim))),funlib["Uref"]["fun"](data.X.reshape((Nx*Nt,xdim)))]).T.reshape((Nx,Nt,2))
            if ramp_projection is not None: print("ramp_projection.shape = {}".format(ramp_projection.shape))
            if collect_flag: 
                _ = self.collect_transition_states(model,data,'daeltime',dirn,num_per_level,num_levels,frac_of_max=frac_of_max,tolerance=2.0,ramp_bounds=[0.01,0.99],ramp_projection=ramp_projection)
                _ = self.collect_transition_states(model,data,'leadtime',dirn,num_per_level,num_levels,frac_of_max=frac_of_max,tolerance=2.0,ramp_bounds=[0.01,0.99],ramp_projection=ramp_projection)
                _ = self.collect_transition_states(model,data,'committor',dirn,num_per_level,num_levels,frac_of_max=frac_of_max,tolerance=0.05,ramp_bounds=[0.05,0.95],ramp_projection=ramp_projection)
            # Plot least-action Uref and max-flux Uref
            time_symbol = 'leadtime'
            negtime = (time_symbol == 'leadtime')
            fig,ax = plt.subplots(ncols=2,nrows=3,figsize=(16,18),sharey='row',sharex='col')
            model.plot_least_action_scalars(self.physical_param_folder,obs_names=["Uref"],fig=fig,ax=[ax[0,0]],negtime=negtime)
            self.plot_maxflux_path(model,data,time_symbol,dirn,num_per_level,num_levels,frac_of_max=frac_of_max,func_key="Uref",fig=fig,ax=ax[0,1])
            ax[0,1].set_xlim(ax[0,0].get_xlim())
            # Now plot the profiles along with the least-action ones
            #fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(16,12),sharey='row',sharex='col')
            func_key_list = ["U","mag"]
            _,_,ims = model.plot_least_action_profiles(self.physical_param_folder,prof_names=func_key_list,fig=fig,ax=ax[1:,0],negtime=negtime)
            for i in range(len(func_key_list)):
                clim = np.array([ims[i].levels[0],ims[i].levels[-1]])
                _,_,imi = self.plot_maxflux_profile(model,data,time_symbol,dirn,num_per_level,num_levels,frac_of_max=frac_of_max,func_key=func_key_list[i],fig=fig,ax=ax[i+1,1],clim=clim)
                ax[i+1,1].set_xlim(ax[i+1,0].get_xlim())
                fig.colorbar(imi,ax=ax[i+1,1])
            # Set x labels to False
            for i in range(ax.shape[0]-1):
                ax[i,0].xaxis.set_visible(False)
                ax[i,1].xaxis.set_visible(False)
            for i in range(ax.shape[0]):
                ax[i,1].yaxis.set_visible(False)
            # Correct position of top row
            pos00 = ax[0,0].get_position()
            pos10 = ax[1,0].get_position()
            ax[0,0].set_position([pos10.x0,pos00.y0,pos10.width,pos00.height])
            pos01 = ax[0,1].get_position()
            pos11 = ax[1,1].get_position()
            ax[0,1].set_position([pos11.x0,pos01.y0,pos11.width,pos01.height])
            fig.savefig(join(self.savefolder,"lap_vs_tpt_ab_profiles"),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
        return
    def collect_transition_states(self,model,data,ramp_name,dirn,num_per_level=5,num_levels=11,frac_of_max=0.9,tolerance=np.inf,ramp_bounds=None,ramp_projection=None):
        # ramp_name can be either committor or leadtime
        # dirn is 'ab' or 'ba'
        # Collect all states down to a given fraction of the maximum
        Nx,Nt,xdim = data.X.shape
        key = list(self.dam_moments.keys())[0]
        fwd_key = 'xb' if dirn=='ab' else 'xa'
        bwd_key = 'ax' if dirn=='ab' else 'bx'
        comm_fwd = self.dam_moments[key][fwd_key][0,:,:]
        comm_bwd = self.dam_moments[key][bwd_key][0,:,:]
        weight = np.ones(data.nshort)/data.nshort #self.chom
        if ramp_name == 'committor':
            symbol = "P_x\\{x\\to B\\}" if dirn=='ab' else "P_x\\{x\\to A\\}"
            ramp = comm_fwd
            if ramp_bounds is None: 
                ramp_min,ramp_max = 0.05,0.95
            else:
                ramp_min,ramp_max = ramp_bounds
            minlevel,maxlevel = 0.0,1.0
        elif ramp_name == 'leadtime':
            symbol = "E_x[\\tau^+|A\\to B]" if dirn=='ab' else "E_x[\\tau^+|B\\to A]"
            eps = 1e-2
            ramp = -self.dam_moments['one'][fwd_key][1,:,:]*(comm_fwd > eps)/(comm_fwd + 1*(comm_fwd < eps))
            ramp[np.where(comm_fwd <= eps)[0]] = np.nan
            print("ramp.shape = {}, min(ramp) = {}, max(ramp) = {}".format(ramp.shape,np.min(ramp),np.max(ramp)))
            minlevel,maxlevel = np.nanmin(ramp),np.nanmax(ramp)
            ramp_min,ramp_max = minlevel + np.array(ramp_bounds)*(maxlevel - minlevel)
            #if ramp_bounds is None:
            #    min_quantile,max_quantile = 0.05,0.95
            #else:
            #    min_quantile,max_quantile = ramp_bounds
            #ramp_min,ramp_max = np.nanquantile(ramp.flatten(),[min_quantile,max_quantile])
        elif ramp_name == 'daeltime':
            symbol = "E_x[\\tau^-|A\\to B]" if dirn=='ab' else "E_x[\\tau^-|B\\to A]"
            eps = 1e-2
            ramp = self.dam_moments['one'][bwd_key][1,:,:]*(comm_bwd > eps)/(comm_bwd + 1*(comm_bwd < eps))
            ramp[np.where(comm_bwd <= eps)[0]] = np.nan
            print("ramp.shape = {}, min(ramp) = {}, max(ramp) = {}".format(ramp.shape,np.min(ramp),np.max(ramp)))
            minlevel,maxlevel = np.nanmin(ramp),np.nanmax(ramp)
            ramp_min,ramp_max = minlevel + np.array(ramp_bounds)*(maxlevel - minlevel)
        # ------------------------- New crazy method: project ramp ---------------------
        if ramp_projection is not None:
            ramp_projection = ramp_projection.reshape((Nx*Nt,ramp_projection.shape[-1]))
            shp,dth,thaxes,_,ramp_proj,_,_,_,bounds = helper.project_field(ramp.flatten(),np.outer(self.chom,np.ones(Nt)).flatten(),ramp_projection)
            print("ramp_proj.shape = {}".format(ramp_proj.shape))
            ii = ((ramp_projection[:,0] - bounds[0,0])/dth[0]).astype(int)
            jj = ((ramp_projection[:,1] - bounds[1,0])/dth[1]).astype(int)
            print("ii.shape = {}, jj.shape = {}".format(ii.shape,jj.shape))
            kk = np.ravel_multi_index((ii,jj),shp)
            print("kk.shape = {}".format(kk.shape))
            ramp = ramp_proj.flat[kk].reshape((Nx,Nt))
            # Check that this ramp really is the right projection
            fig,ax = helper.plot_field_2d(ramp[:,0],self.chom,ramp_projection.reshape((Nx,Nt,ramp_projection.shape[-1]))[:,0],fun0name="magref",fun1name="Uref")
            fig.savefig(join(self.savefolder,"rampcheck_{}".format(ramp_name)))
            plt.close(fig)
        # ------------------------------------------------------------------------------
        levels = np.linspace(ramp_min,ramp_max,num_levels)
        tolerance = min(tolerance,np.abs(ramp_max-ramp_min)/(2*num_levels))
        rflux_idx = [[] for i in range(num_levels)]
        rflux = [[] for i in range(num_levels)]
        #rflux_idx = -np.ones((len(levels),num_per_level),dtype=int) # -1 is a filler
        # TODO: build flux_idx one level at a time to allow for missing levels and stop crashing.
        for i in range(num_levels):
            new_idx,new_rflux,_ = self.maximize_rflux_on_surface(model,data,ramp.reshape((Nx,Nt,1)),comm_bwd,comm_fwd,weight,levels[i],tolerance,num_per_level,frac_of_max)
            rflux_idx[i] += new_idx
            rflux[i] += new_rflux
        # Save a dictionary with the levels and indices
        flux_dict = dict({
            "levels": levels,
            "idx": rflux_idx,
            "rflux": rflux,
            "minlevel": minlevel,
            "maxlevel": maxlevel,
            "symbol": symbol,
            })
        pickle.dump(flux_dict,open((join(self.savefolder,"flux_{}_{}_fom{}_nlev{}_nplev{}".format(ramp_name,dirn,frac_of_max,num_levels,num_per_level))).replace(".","p"),"wb"))
        return rflux_idx
    def plot_flux_distributions_1d_driver(self,model,data):
        funlib = model.observable_function_library()
        Nx,Nt,xdim = data.X.shape
        ramp_abbrv_list = ["Uref"] #,"Uref","Uref","Uref"]
        func_abbrv_list = ["vTintref"] #,"vTinttop","U67"]
        for i in range(len(ramp_abbrv_list)):
            ramp_abbrv = ramp_abbrv_list[i]
            func_abbrv = func_abbrv_list[i]
            ramp = funlib[ramp_abbrv]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            ramp_name = funlib[ramp_abbrv]["name"]
            ramp_units = funlib[ramp_abbrv]["units"]
            ramp_unit_symbol = funlib[ramp_abbrv]["unit_symbol"]
            func = funlib[func_abbrv]["fun"](data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            func_name = funlib[func_abbrv]["name"]
            func_units = funlib[func_abbrv]["units"]
            func_unit_symbol = funlib[func_abbrv]["unit_symbol"]
            fig,ax = self.plot_flux_distributions_1d_compact(model,data,ramp,ramp_name,ramp_units,ramp_unit_symbol,func,func_name,func_units,func_unit_symbol,num_levels=4)
            fig.savefig(join(self.savefolder,"flux_dist_ramp{}_func{}".format(ramp_abbrv,func_abbrv)),bbox_inches="tight",pad_inches=0.2)
            print("Just saved a flux dist fig")
            plt.close(fig)
        return
    def plot_flux_distributions_1d_compact(self,model,data,ramp,ramp_name,ramp_units,ramp_unit_symbol,func,func_name,func_units,func_unit_symbol,num_levels=5,frac_of_max=0.0,fig=None,ax=None,clim=None):
        # At each level set of ramp, plot a distribution of flux density across the other variable func. (Will later extend to 2d). 
        # The observable should be correlated with the committor...
        # Add a second curve, dotted, for A->A phase

        max_num_states = None
        comm_fwd = self.dam_moments['one']['xb'][0,:,:]
        comm_bwd = self.dam_moments['one']['ax'][0,:,:]
        eps = 1e-6
        interior_idx = np.where((comm_fwd > eps)*(comm_fwd < 1-eps))
        ramp_min = np.nanmin(ramp[interior_idx])
        ramp_max = np.nanmax(ramp[interior_idx])
        func_min = np.nanmin(func[interior_idx])
        func_max = np.nanmax(func[interior_idx])
        print("ramp_min = {}, ramp_max = {}".format(ramp_min,ramp_max))
        ramp_comm_corr = np.nanmean((ramp - np.nanmean(ramp))*(comm_fwd - np.nanmean(comm_fwd)))
        print("ramp_comm_corr = {}".format(ramp_comm_corr))
        dramp = (ramp_max - ramp_min)/num_levels
        ramp_levels = np.linspace(ramp_min+0.5*dramp,ramp_max-0.5*dramp,num_levels)
        if ramp_comm_corr < 0:
            ramp_levels = ramp_levels[::-1]
        ramp_tol = np.min(np.abs(np.diff(ramp_levels)))/4
        print("ramp_levels = {}".format(ramp_levels))
        weight = self.chom
        Nx,Nt,xdim = data.X.shape
        if fig is None or ax is None:
            fig,ax = plt.subplots(nrows=2,figsize=(6,12),sharey=False,sharex=True)
        num_bins = 15
        ax[0].set_title(r"$A\to B$",fontdict=font)
        ax[1].set_title(r"$B\to A$",fontdict=font)
        #ax[2].set_title(r"$A\to A$",fontdict=font)
        #ax[3].set_title(r"$B\to B$",fontdict=font)
        dirn_list = ['ab','ba','aa','bb'][:2]
        # Make a list with the signs and offsets
        # Normalize by rate
        rate = self.dam_moments['one']['rate_avg'][0]
        print(f"rate = {rate}")
        for di in range(2): #ab,ba,aa,bb
            handles = []
            ax[di].axhline(0,color='black')
            ax[di].set_ylabel(r"Flux density [(%s)$^{-1}\cdot$days$^{-1}$]"%(func_unit_symbol))
            for ri in range(len(ramp_levels)):
                comm_fwd_di = comm_fwd*(dirn_list[di].endswith('b')) + (1-comm_fwd)*(dirn_list[di].endswith('a'))
                comm_bwd_di = comm_bwd*(dirn_list[di].startswith('a')) + (1-comm_bwd)*(dirn_list[di].startswith('b'))
                ridx,rflux,_ = self.maximize_rflux_on_surface(model,data,ramp.reshape((Nx,Nt,1)),comm_bwd_di,comm_fwd_di,weight,ramp_levels[ri],ramp_tol,max_num_states,frac_of_max)
                tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
                hist,bin_edges = np.histogram(func[ridx,tidx],weights=rflux,density=False, range=(func_min,func_max), bins=num_bins)
                rate_di = np.nansum(hist*np.diff(bin_edges)) * np.sign(ramp_comm_corr)
                normalizer = np.abs(rate/rate_di)
                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                color_frac=(ramp_levels[ri]-ramp_min)/(ramp_max-ramp_min)
                if ramp_comm_corr < 0: color_frac = 1-color_frac
                h, = ax[di].plot(bin_centers*func_units,hist*normalizer,color=plt.cm.coolwarm(color_frac),marker='o',label=r"%s$=$%.1f %s"%(ramp_name,ramp_levels[ri]*ramp_units,ramp_unit_symbol),linestyle='-',linewidth=2.5)
                handles += [h]
        ax[0].legend(handles=handles,loc='lower right')
        ax[-1].set_xlabel("%s [%s]"%(func_name,func_unit_symbol), fontdict=medfont)
        return fig,ax
    def plot_flux_distributions_1d(self,model,data,ramp,ramp_name,ramp_units,ramp_unit_symbol,func,func_name,func_units,func_unit_symbol,num_levels=5,frac_of_max=0.0,fig=None,ax=None,clim=None):
        # At each level set of ramp, plot a distribution of flux density across the other variable func. (Will later extend to 2d). 
        # The observable should be correlated with the committor...
        # Add a second curve, dotted, for A->A phase

        max_num_states = None
        comm_fwd = self.dam_moments['one']['xb'][0,:,:]
        comm_bwd = self.dam_moments['one']['ax'][0,:,:]
        eps = 1e-6
        interior_idx = np.where((comm_fwd > eps)*(comm_fwd < 1-eps))
        ramp_min = np.nanmin(ramp[interior_idx])
        ramp_max = np.nanmax(ramp[interior_idx])
        func_min = np.nanmin(func[interior_idx])
        func_max = np.nanmax(func[interior_idx])
        print("ramp_min = {}, ramp_max = {}".format(ramp_min,ramp_max))
        ramp_comm_corr = np.nanmean((ramp - np.nanmean(ramp))*(comm_fwd - np.nanmean(comm_fwd)))
        print("ramp_comm_corr = {}".format(ramp_comm_corr))
        dramp = (ramp_max - ramp_min)/num_levels
        ramp_levels = np.linspace(ramp_min+0.5*dramp,ramp_max-0.5*dramp,num_levels)
        if ramp_comm_corr < 0:
            ramp_levels = ramp_levels[::-1]
        ramp_tol = np.min(np.abs(np.diff(ramp_levels)))/4
        print("ramp_levels = {}".format(ramp_levels))
        weight = self.chom
        Nx,Nt,xdim = data.X.shape
        if fig is None or ax is None:
            fig,ax = plt.subplots(nrows=num_levels,ncols=4,figsize=(16,4*num_levels),sharey="col",sharex=True)
        num_bins = 15
        for ri in range(len(ramp_levels)):
            ax[ri,0].set_ylabel("%s = %.2f %s"%(ramp_name,ramp_levels[ri]*ramp_units,ramp_unit_symbol),fontdict=medfont)
            # A -> A
            ax[ri,0].axhline(0,color='black')
            ridx,rflux,_ = self.maximize_rflux_on_surface(model,data,ramp.reshape((Nx,Nt,1)),comm_bwd,1-comm_fwd,weight,ramp_levels[ri],ramp_tol,max_num_states,frac_of_max)
            tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
            hist,bin_edges = np.histogram(func[ridx,tidx],weights=rflux,density=False, range=(func_min,func_max), bins=num_bins)
            rate_aa = np.nansum(hist*np.diff(bin_edges)) * np.sign(ramp_comm_corr)
            normalizer = 1.0 #np.max(np.abs(hist))
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            h, = ax[ri,0].plot(bin_centers*func_units,hist/normalizer,color='deepskyblue',marker='o',label=r"$A\to A$",linestyle='-')
            #ax[ri,0].legend(handles=[h])
            ax[0,0].set_title(r"$A\to A$",fontdict=font)
            #if ri == 0: ax[ri,0].set_ylabel("Flux density")
            # A -> B
            ax[ri,1].axhline(0,color='black')
            ridx,rflux,_ = self.maximize_rflux_on_surface(model,data,ramp.reshape((Nx,Nt,1)),comm_bwd,comm_fwd,weight,ramp_levels[ri],ramp_tol,max_num_states,frac_of_max)
            tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
            hist,bin_edges = np.histogram(func[ridx,tidx],weights=rflux,density=False, range=(func_min,func_max), bins=num_bins)
            rate_ab = np.nansum(hist*np.diff(bin_edges)) * np.sign(ramp_comm_corr)
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            normalizer = 1.0 #np.max(np.abs(hist))
            h, = ax[ri,1].plot(bin_centers*func_units,hist/normalizer,color='darkorange',marker='o',label=r"$A\to B$",linestyle='-')
            #ax[ri,1].legend(handles=[h])
            ax[0,1].set_title(r"$A\to B$", fontdict=font)
            #if ri == 0: ax[ri,1].set_ylabel("Flux density")
            # B -> B
            ax[ri,2].axhline(0,color='black')
            ridx,rflux,_ = self.maximize_rflux_on_surface(model,data,ramp.reshape((Nx,Nt,1)),1-comm_bwd,comm_fwd,weight,ramp_levels[ri],ramp_tol,max_num_states,frac_of_max)
            tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
            hist,bin_edges = np.histogram(func[ridx,tidx],weights=rflux,density=False, range=(func_min,func_max), bins=num_bins)
            rate_aa = np.nansum(hist*np.diff(bin_edges)) * np.sign(ramp_comm_corr)
            normalizer = 1.0 #np.max(np.abs(hist))
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            h, = ax[ri,2].plot(bin_centers*func_units,hist/normalizer,color='red',marker='o',label=r"$B\to B$",linestyle='-')
            #ax[ri,2].legend(handles=[h])
            ax[0,2].set_title(r"$B\to B$", fontdict=font)
            #if ri == 0: ax[ri,2].set_ylabel("Flux density")
            # B -> A
            ax[ri,3].axhline(0,color='black')
            ridx,rflux,_ = self.maximize_rflux_on_surface(model,data,ramp.reshape((Nx,Nt,1)),1-comm_bwd,1-comm_fwd,weight,ramp_levels[ri],ramp_tol,max_num_states,frac_of_max)
            tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
            hist,bin_edges = np.histogram(func[ridx,tidx],weights=rflux,density=False, range=(func_min,func_max), bins=num_bins)
            rate_ba = np.nansum(hist*np.diff(bin_edges)) * np.sign(ramp_comm_corr)
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            normalizer = 1.0 #np.max(np.abs(hist))
            h, = ax[ri,3].plot(bin_centers*func_units,hist/normalizer,color='mediumspringgreen',marker='o',label=r"$B\to A$",linestyle='-')
            #ax[ri,3].legend(handles=[h])
            ax[0,3].set_title(r"$B\to A$",fontdict=font)
            #if ri == 0: ax[ri,3].set_ylabel("Flux density")
            for j in range(4):
                ax[ri,j].yaxis.set_ticklabels([])
                if ri == len(ramp_levels) - 1: 
                    ax[ri,j].set_xlabel("%s [%s]"%(func_name,func_unit_symbol), fontdict=medfont)
        return fig,ax
    def plot_transition_states(self,model,data,ramp_name,dirn,num_per_level=10,num_levels=3,frac_of_max=0.9,func_key="U",plot_level_subset=None):
        # func is now an altitude-dependent function
        if plot_level_subset is None: plot_level_subset = np.arange(num_levels)
        if len(plot_level_subset) > num_levels: sys.exit("ERROR: plot_level_subset = {} while num_levels = {}".format(plot_level_subset,num_levels))
        #flux_dict = pickle.load(open(join(self.savefolder,"flux_{}_{}_nlev{}_nplev{}".format(ramp_name,dirn,num_levels,num_per_level)),"rb"))
        flux_dict = pickle.load(open((join(self.savefolder,"flux_{}_{}_fom{}_nlev{}_nplev{}".format(ramp_name,dirn,frac_of_max,num_levels,num_per_level))).replace(".","p"),"rb"))
        rflux_idx = flux_dict["idx"]
        levels = flux_dict["levels"]
        levels_nondegenerate = []
        rflux = flux_dict["rflux"]
        if len(rflux_idx) != len(rflux):
            raise Exception("DOH: len(rflux_idx) = {}, len(rflux) = {}".format(len(rflux_idx),len(rflux)))
        minlevel,maxlevel = flux_dict["minlevel"],flux_dict["maxlevel"]
        cmap = plt.cm.coolwarm if dirn=='ab' else plt.cm.coolwarm_r
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current_display/2))
        # Get the colors in order
        num_total_states = 0
        colorlist = []
        labellist = []
        labellist_unique = []
        colorlist_unique = []
        real_levels = [] #np.zeros((num_levels,num_per_level))
        rflux_idx_flat = []
        print("levels = {}, minlevel = {}, maxlevel = {}".format(levels,minlevel,maxlevel))
        funlib = model.observable_function_library()
        comm_fwd = self.dam_moments['one']['xb'][0,:,0]
        for i in range(len(plot_level_subset)):
            plsi = plot_level_subset[i]
            Uref = funlib["Uref"]["fun"](data.X[rflux_idx[plsi],tidx])
            #print("dirn={}, func_key={}, plsi = {}. committor range = {},{}. Uref range = {},{}".format(dirn,func_key,plsi,np.min(comm_fwd[rflux_idx[plsi]]),np.max(comm_fwd[rflux_idx[plsi]]),np.min(Uref),np.max(Uref)))
            #print("rflux_idx[plsi] = {}".format(rflux_idx[plsi]))
            numi = len(rflux_idx[plsi])
            num_total_states += numi
            rflux_idx_flat += rflux_idx[plsi]
            norm_level = (levels[plsi]-minlevel)/(maxlevel-minlevel)
            color = cmap(norm_level) if norm_level != 0.5 else 'gold'
            colorlist_unique += [color]
            colorlist += [color for j in range(numi)]
            if numi > 0: 
                labellist_unique += [r"$%s=%.1f$"%(flux_dict["symbol"],levels[plsi])]
                labellist += [labellist_unique[-1]] + ["" for j in range(numi-1)]
                levels_nondegenerate += [levels[plsi]]
            real_levels += [levels[plsi] for j in range(numi)] #[i,:] = levels[plot_level_subset[i]]
        zorderlist = np.random.permutation(np.arange(num_total_states))
        # DO NOT plot all the lines. 
        #fig,ax = model.plot_multiple_states(data.X[rflux_idx_flat,tidx],real_levels,ramp_name,colorlist=colorlist,zorderlist=zorderlist,key=func_key,labellist=labellist)
        #title = r"$A\to B$ transition states" if dirn=='ab' else r"$B\to A$ transition states"
        #ax.set_title(title,fontdict=font)
        #fig.savefig(join(self.savefolder,"trans_states_plot_{}_{}_nlev{}_nplev{}_funckey{}".format(ramp_name,dirn,num_levels,num_per_level,func_key)),bbox_inches="tight",pad_inches=0.2)
        #plt.close(fig)
        # Plot mean and standard deviation
        #fig,ax = model.plot_state_distribution(data.X[:,tidx],rflux,rflux_idx,levels[plot_level_subset],ramp_name,colors=colorlist_unique,key=func_key,labels=labellist_unique)
        print("Before calling plot_state_distribution: len(rflux) = %i, len(rflux_idx) = %i, len(levels_nondegenerate) = %i, len(colorlist_unique) = %i, len(labellist_unique) = %i"%(len(rflux),len(rflux_idx),len(levels_nondegenerate),len(colorlist_unique),len(labellist_unique)))
        fig,ax = model.plot_state_distribution(data.X[:,tidx],rflux,rflux_idx,levels_nondegenerate,ramp_name,colors=colorlist_unique,key=func_key,labels=labellist_unique)
        fig.savefig(join(self.savefolder,("trans_state_dist_plot_{}_{}_fom{}_nlev{}_nplev{}_funckey{}".format(ramp_name,dirn,frac_of_max,num_levels,num_per_level,func_key)).replace(".","p")),bbox_inches="tight",pad_inches=0.2)
        print("Saved transition state distribution in {}".format(self.savefolder))
        plt.close(fig)
        return
