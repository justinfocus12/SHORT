# This file collects stuff specific to TPT and the Holton-Mass model
import numpy as np
from numpy import save,load
import scipy
from scipy.stats import describe
from scipy import special
from sklearn import linear_model
import matplotlib
matplotlib.use('AGG')
#matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['savefig.bbox'] = 'tight'
#matplotlib.rcParams['savefig.pad_inches'] = 0
smallfont = {'family': 'serif', 'size': 8}
font = {'family': 'serif', 'size': 18}
ffont = {'family': 'serif', 'size': 25}
bigfont = {'family': 'serif', 'size': 30}
bbigfont = {'family': 'serif', 'size': 40}
giantfont = {'family': 'serif', 'size': 80}
ggiantfont = {'family': 'serif', 'size': 120}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
        return ab_starts,ab_ends,ba_starts,ba_ends,self.dam_emp
    def compile_data(self,model):
        print("In TPT: self.nshort = {}".format(self.nshort))
        t_short,x_short = model.load_short_traj(self.short_simfolder,self.nshort)
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
    def plot_field_long(self,model,data,field,fieldname,field_abb,field_fun=None,units=1.0,tmax=70,field_unit_symbol=None,time_unit_symbol=None,include_reactive=True):
        print("Beginning plot field long")
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
        if ab_starts[0] > ab_ends[0]: ab_ends = ab_ends[1:]
        if ab_starts[-1] > ab_ends[-1]: ab_starts = ab_starts[:-1]
        if ba_starts[0] > ba_ends[0]: ba_ends = ba_ends[1:]
        if ba_starts[-1] > ba_ends[-1]: ba_starts = ba_starts[:-1]
        # Plot, marking transitions
        tmax = min(t_long[-1],tmax)
        # Interpolate field
        timax = np.argmin(np.abs(t_long-tmax))
        tsubset = np.linspace(0,timax-1,min(timax,5000)).astype(int)
        if field_fun is None:
            field_long = self.out_of_sample_extension(field,data,x_long[tsubset])
            ab_long = self.out_of_sample_extension(field,data,model.tpt_obs_xst)
        else:
            field_long = field_fun(x_long[tsubset]).flatten()
            ab_long = field_fun(model.tpt_obs_xst).flatten()
        print("field_long.shape = {}".format(field_long.shape))
        fig,ax = plt.subplots(ncols=2,figsize=(22,7),gridspec_kw={'width_ratios': [3,1]},tight_layout=True,sharey=True)
        ax[0].plot(t_long[tsubset],units*field_long,color='black')
        ax[0].plot(t_long[[tsubset[0],tsubset[-1]]],ab_long[0]*np.ones(2)*units,color='skyblue',linewidth=2.5)
        ax[0].plot(t_long[[tsubset[0],tsubset[-1]]],ab_long[1]*np.ones(2)*units,color='red',linewidth=2.5)
        dthab = np.abs(ab_long[1]-ab_long[0])
        ax[0].text(0,units*(ab_long[0]+0.01*dthab),asymb,fontdict=bbigfont,color='black',weight='bold')
        ax[0].text(0,units*(ab_long[1]+0.01*dthab),bsymb,fontdict=bbigfont,color='black',weight='bold')
        for i in range(len(ab_starts)):
            if ab_ends[i] < timax:
                ax[0].axvspan(t_long[ab_starts[i]],t_long[ab_ends[i]],facecolor='orange',alpha=0.5,zorder=-1)
        for i in range(len(ba_starts)):
            if ba_ends[i] < timax:
                ax[0].axvspan(t_long[ba_starts[i]],t_long[ba_ends[i]],facecolor='mediumspringgreen',alpha=0.5,zorder=-1)
        xlab = "Time"
        if time_unit_symbol is not None: xlab += " ({})".format(time_unit_symbol)
        ax[0].set_xlabel(xlab,fontdict=bigfont)
        ylab = fieldname
        if field_unit_symbol is not None: ylab += " ({})".format(field_unit_symbol)
        ax[0].set_ylabel(ylab,fontdict=bigfont)
        #ax.set_title("Long integration",fontdict=font)
        ax[0].tick_params(axis='both',labelsize=25)
        #ax.yaxis.set_major_locator(ticker.NullLocator())
        # Now plot the densities in y
        self.display_1d_densities(model,data,[field_abb],'vertical',fig=fig,ax=ax[1],include_reactive=include_reactive)
        ax[1].yaxis.set_visible(False)
        ax[1].set_xlabel("Probability density",fontdict=bigfont)
        ax[1].tick_params(axis='both',labelsize=25)
        fig.savefig(join(self.savefolder,"{}_long".format(field_abb)),bbox_inches="tight",pad_inches=0.2)
        del x_long
        plt.close(fig)
        print("Done plotting field long")
        return
    def plot_field_long_2d(self,model,data,fieldnames,field_funs,field_abbs,units=[1.0,1.0],tmax=70,field_unit_symbols=["",""],orientation=None):
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
        if ab_starts[0] > ab_ends[0]: ab_ends = ab_ends[1:]
        if ab_starts[-1] > ab_ends[-1]: ab_starts = ab_starts[:-1]
        if ba_starts[0] > ba_ends[0]: ba_ends = ba_ends[1:]
        if ba_starts[-1] > ba_ends[-1]: ba_starts = ba_starts[:-1]
        timax = np.argmin(np.abs(t_long-tmax))
        print("t_long[timax] = {}".format(t_long[timax]))
        tsubset = np.linspace(0,timax-1,min(timax,15000)).astype(int)
        # Plot the two fields vs. each other, marking transitions
        field0 = field_funs[0](x_long[tsubset]).flatten()
        field1 = field_funs[1](x_long[tsubset]).flatten()
        ab0 = field_funs[0](model.tpt_obs_xst).flatten()
        ab1 = field_funs[1](model.tpt_obs_xst).flatten()
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
        ax.set_xlabel(r"%s (%s)"%(fieldnames[0],field_unit_symbols[0]),fontdict=ffont)
        ax.set_ylabel(r"%s (%s)"%(fieldnames[1],field_unit_symbols[1]),fontdict=ffont)
        ax.tick_params(axis='both',labelsize=20)
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
    def plot_projections_1d_array(self,model,data):
        q = model.q
        funlib = model.observable_function_library()
        theta1d_list = ['Uref','vTintref'] #,'LASSO']
        Nth = len(theta1d_list)
        n = q['Nz']-1
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
    def compute_dam_moments_abba_finlag(self,model,data,function):
        Nx,Nt,xdim = data.X.shape
        dam_keys = list(model.dam_dict.keys())
        num_bvp = len(dam_keys)
        num_moments = self.num_moments
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
        bdy_dist_x = np.minimum(adist_x,bdist_x)
        ramp_ab = Fp[0,0,:,:] #adist_x / (adist_x + bdist_x)
        ramp_ba = 1-Fp[0,0,:,:] #Fm[0,0,:,:] #bdist_x / (adist_x + bdist_x)
        #data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_seq[-1])
        #data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current)
        MFp = np.zeros((2*num_bvp,num_moments+1,Nx))
        #fd_total_decay = 0.5
        #fd_weights = fd_total_decay**((np.arange(len(self.lag_time_seq))-1)/(len(self.lag_time_seq)-1))
        #print("fd_weights = {}".format(fd_weights))
        #fd_decay_rate = fd_total_decay**(1/(len(self.lag_time_seq)-1))
        # TODO: figure this the heck out
        for i in range(num_moments+1):
            for j in range(len(self.lag_time_seq)-1):
                data.insert_boundaries_fwd(bdy_dist_x,data.t_x[j],data.t_x[-1])
                data.insert_boundaries_bwd(bdy_dist_x,data.t_x[j],data.t_x[0])
                MFp[:num_bvp,i,:] += Fp[:num_bvp,i,np.arange(data.nshort),data.first_exit_idx_fwd]*Fm[:num_bvp,i,np.arange(data.nshort),data.first_exit_idx_bwd]*(ramp_ab[:,j+1] - ramp_ab[:,j])
                MFp[num_bvp:,i,:] += Fp[num_bvp:,i,np.arange(data.nshort),data.first_exit_idx_fwd]*Fm[num_bvp:,i,np.arange(data.nshort),data.first_exit_idx_bwd]*(ramp_ba[:,j+1] - ramp_ba[:,j])
                #MFp[:num_bvp,i,:] += (ramp_ab[:,j]*Fp[:num_bvp,i,:,j] - ramp_ab[:,0]*Fp[:num_bvp,i,:,0])/self.lag_time_seq[j] * (j<=data.first_exit_idx) * fd_weights[j]
                #MFp[num_bvp:,i,:] += (ramp_ba[:,j]*Fp[num_bvp:,i,:,j] - ramp_ba[:,0]*Fp[num_bvp:,i,:,0])/self.lag_time_seq[j] * (j<=data.first_exit_idx) * fd_weights[j]
                #normalizer += fd_weights[j]*(j<=data.first_exit_idx)
            #print("normalizer: shp={}, min={}, max={}, mean={}, std={}".format(normalizer.shape,np.min(normalizer),np.max(normalizer),np.mean(normalizer),np.std(normalizer)))
            MFp[:,i,:] *= 1.0/self.lag_time_seq[-1]
            #if i > 0:
            #    MFp[:num_bvp,i] += ramp_ab[:,0]*i*Pay[:num_bvp,:,0]*Fp[:num_bvp,i-1,:,0]
            #    MFp[num_bvp:,i] += ramp_ba[:,0]*i*Pay[num_bvp:,:,0]*Fp[num_bvp:,i-1,:,0]
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
    def plot_lifecycle_correlations(self,model,keys=None):
        # After they've been computed, plot
        if keys is None: keys = list(model.corr_dict.keys())
        names = [r"$A\to A$",r"$A\to B$",r"$B\to B$",r"$B\to A$"]
        # Correlations
        maxcorr = 0
        fig,ax = plt.subplots(nrows=len(keys),figsize=(6,3*len(keys)),tight_layout=True,sharex=True)
        for k in range(len(keys)):
            print("key = {}. corr_dga range = ({},{}). corr_emp range = ({},{})".format(keys[k],np.min(self.lifecycle_corr_dga[keys[k]]),np.max(self.lifecycle_corr_dga[keys[k]]),np.min(self.lifecycle_corr_emp[keys[k]]),np.max(self.lifecycle_corr_emp[keys[k]])))
            ax[k].plot(names,self.lifecycle_corr_emp[keys[k]],marker='o',color='black')
            ax[k].plot(names,self.lifecycle_corr_dga[keys[k]],marker='o',color='red')
            ax[k].set_title(r"$\Gamma = $%s"%model.corr_dict[keys[k]]['name'])
            ax[k].set_ylabel(r"Corr($\Gamma,q^+q^-$)")
            #ax[k].set_ylim([-1,1])
            ax[k].plot(names,np.zeros(len(names)),linestyle='--',color='black')
            maxcorr = max(maxcorr,max(np.max(np.abs(self.lifecycle_corr_dga[keys[k]])),np.max(np.abs(self.lifecycle_corr_emp[keys[k]]))))
        for k in range(len(keys)):
            ax[k].set_ylim([-maxcorr,maxcorr])
        #fig.suptitle("Lifecycle correlations")
        fig.savefig(join(self.savefolder,"lifecycle_corr"))
        plt.close(fig)
        # Means
        fig,ax = plt.subplots(nrows=len(keys),figsize=(6,3*len(keys)),tight_layout=True,sharex=True)
        for k in range(len(keys)):
            ax[k].plot(names,self.lifecycle_mean_emp[keys[k]],marker='o',color='black')
            ax[k].plot(names,self.lifecycle_mean_dga[keys[k]],marker='o',color='red')
            ax[k].set_title(r"$\Gamma = $%s"%model.corr_dict[keys[k]]['name'])
            ax[k].set_ylabel(r"$\langle\Gamma,q^+q^-\rangle_\pi")
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
        self.lifecycle_corr_dga = {} #np.zeros((len(keys),4))
        self.lifecycle_corr_emp = {} #np.zeros((len(keys),4)) 
        self.lifecycle_mean_dga = {} #np.zeros((len(keys),4))
        self.lifecycle_mean_emp = {} #np.zeros((len(keys),4)) 
        for k in range(len(keys)):
            self.lifecycle_corr_dga[keys[k]] = np.zeros(4)
            self.lifecycle_corr_emp[keys[k]] = np.zeros(4)
            self.lifecycle_mean_dga[keys[k]] = np.zeros(4)
            self.lifecycle_mean_emp[keys[k]] = np.zeros(4)
            print("data.X.shape = {}".format(data.X.shape))
            Pay = model.corr_dict[keys[k]]['pay'](data.X[:,0,:]).flatten()
            print("Pay.shape = {}".format(Pay.shape))
            t_long,x_long = model.load_long_traj(self.long_simfolder)
            Pay_long = model.corr_dict[keys[k]]['pay'](x_long).flatten()
            del x_long
            print("Pay_long.shape = {}".format(Pay_long.shape))
            f.write("Correlation function %s\n"%model.corr_dict[keys[k]]['name'])
            # A -> B
            f.write("\tA->B: ")
            comm_bwd = self.dam_moments[dk0]['ax'][0,:,0]
            comm_fwd = self.dam_moments[dk0]['xb'][0,:,0]
            reactive_flag = 1*(self.long_from_label==-1)*(self.long_to_label==1)
            mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)
            corr_dga = (mean_trans_dga - np.sum(self.chom*comm_bwd*comm_fwd)*np.sum(self.chom*Pay))/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
            f.write("DGA: mean = %3.3e, corr = %3.3e, "%(mean_trans_dga,corr_dga))
            mean_trans_emp = np.mean(reactive_flag*Pay_long)
            corr_emp = (mean_trans_emp - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
            f.write("EMP: mean = %3.3e, corr = %3.3e\n"%(mean_trans_emp,corr_emp))
            self.lifecycle_corr_dga[keys[k]][1] = corr_dga #
            self.lifecycle_corr_emp[keys[k]][1] = corr_emp #
            self.lifecycle_mean_dga[keys[k]][1] = mean_trans_dga #
            self.lifecycle_mean_emp[keys[k]][1] = mean_trans_emp #
            # B -> A
            f.write("\tB->A: ")
            comm_bwd = self.dam_moments[dk0]['bx'][0,:,0]
            comm_fwd = self.dam_moments[dk0]['xa'][0,:,0]
            reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==-1)
            mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)
            corr_dga = (mean_trans_dga - np.sum(self.chom*comm_bwd*comm_fwd)*np.sum(self.chom*Pay))/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
            f.write("DGA: mean = %3.3e, corr = %3.3e, "%(mean_trans_dga,corr_dga))
            mean_trans_emp = np.mean(reactive_flag*Pay_long)
            corr_emp = (mean_trans_emp - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
            f.write("EMP: mean = %3.3e, corr = %3.3e\n"%(mean_trans_emp,corr_emp))
            self.lifecycle_corr_dga[keys[k]][3] = corr_dga #[k,3] = corr_dga
            self.lifecycle_corr_emp[keys[k]][3] = corr_emp #[k,3] = corr_emp
            self.lifecycle_mean_dga[keys[k]][3] = mean_trans_dga #
            self.lifecycle_mean_emp[keys[k]][3] = mean_trans_emp #
            # A -> A
            f.write("\tA->A: ")
            comm_bwd = self.dam_moments[dk0]['ax'][0,:,0]
            comm_fwd = self.dam_moments[dk0]['xa'][0,:,0]
            reactive_flag = 1*(self.long_from_label==-1)*(self.long_to_label==-1)
            mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)
            corr_dga = (mean_trans_dga - np.sum(self.chom*comm_bwd*comm_fwd)*np.sum(self.chom*Pay))/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
            f.write("DGA: mean = %3.3e, corr = %3.3e, "%(mean_trans_dga,corr_dga))
            mean_trans_emp = np.mean(reactive_flag*Pay_long)
            corr_emp = (mean_trans_emp - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
            f.write("EMP: mean = %3.3e, corr = %3.3e\n"%(mean_trans_emp,corr_emp))
            self.lifecycle_corr_dga[keys[k]][0] = corr_dga
            self.lifecycle_corr_emp[keys[k]][0] = corr_emp
            self.lifecycle_mean_dga[keys[k]][0] = mean_trans_dga #
            self.lifecycle_mean_emp[keys[k]][0] = mean_trans_emp #
            # B -> B
            f.write("\tB->B: ")
            comm_bwd = self.dam_moments[dk0]['bx'][0,:,0]
            comm_fwd = self.dam_moments[dk0]['xb'][0,:,0]
            reactive_flag = 1*(self.long_from_label==1)*(self.long_to_label==1)
            mean_trans_dga = np.sum(self.chom*comm_bwd*comm_fwd*Pay)
            corr_dga = (mean_trans_dga - np.sum(self.chom*comm_bwd*comm_fwd)*np.sum(self.chom*Pay))/np.sqrt(np.sum(self.chom*(comm_bwd*comm_fwd)**2)*np.sum(self.chom*Pay**2))
            f.write("DGA: mean = %3.3e, corr = %3.3e, "%(mean_trans_dga,corr_dga))
            mean_trans_emp = np.mean(reactive_flag*Pay_long)
            corr_emp = (mean_trans_emp - np.mean(reactive_flag)*np.mean(Pay_long))/np.sqrt(np.mean(reactive_flag**2)*np.mean(Pay_long**2))
            f.write("EMP: mean = %3.3e, corr = %3.3e\n"%(mean_trans_emp,corr_emp))
            self.lifecycle_corr_dga[keys[k]][2] = corr_dga
            self.lifecycle_corr_emp[keys[k]][2] = corr_emp
            self.lifecycle_mean_dga[keys[k]][2] = mean_trans_dga #
            self.lifecycle_mean_emp[keys[k]][2] = mean_trans_emp #
        return
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
            self.dam_moments[keys[k]]['rate_avg'] = 0.5*(self.dam_moments[keys[k]]['rate_ab'] + self.dam_moments[keys[k]]['rate_ba'])
            f.write("Damage function %s\n"%model.dam_dict[keys[k]]['name_full'])
            f.write("\tA->B\n")
            dga_rate = self.dam_moments[keys[k]]['rate_avg'][0] #['rate_ab'][0]
            emp_rate = len(self.dam_emp[keys[k]]['ab'])/(t_long[-1] - t_long[0])
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
                hemp, = ax.plot(units_k*eg_emp*np.ones(2),units_t*(et_emp + np.sqrt(vt_emp)*np.array([-1,1])), color='black', linestyle='-',label='Empirical')
                ax.plot(units_k*(eg_emp + np.sqrt(vg_emp)*np.array([-1,1])), units_t*et_emp*np.ones(2), color='black', linestyle='-',label='Empirical')
                hdga, = ax.plot(units_k*eg_dga*np.ones(2),units_t*(et_dga + np.sqrt(vt_dga)*np.array([-1,1])), color='red', linestyle='-',label='DGA')
                ax.plot(units_k*(eg_dga + np.sqrt(vg_dga)*np.array([-1,1])), units_t*et_dga*np.ones(2), color='red', linestyle='-',label='DGA')
                ax.set_xlabel(r"$%s (%s)$"%(model.dam_dict[keys[k]]['name_full'],unit_symbol_k),fontdict=font)
                ax.set_ylabel(r"$%s (%s)$"%(model.dam_dict['one']['name_full'],unit_symbol_t),fontdict=font)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                ax.set_title(r"Empirical correlations $A\to B$")
                ax.legend(handles=[hemp,hdga])
                fig.savefig(join(self.savefolder,"corr{}_ab".format(keys[k])))
                plt.close(fig)
                f.write("\t\tCorrelation with T: DGA: %3.3e, EMP: %3.3e\n"%(dga_corr,emp_corr))
            f.write("\t\tRate: DGA: %3.3e, EMP: %3.3e\n"%(dga_rate,emp_rate))
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
            f.write("\tB->A\n")
            dga_rate = self.dam_moments[keys[k]]['rate_avg'][0] #['rate_ba'][0]
            emp_rate = len(self.dam_emp[keys[k]]['ba'])/(t_long[-1] - t_long[0])
            # Compute the correlation with T
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
            hemp, = ax.plot(units_k*eg_emp*np.ones(2),units_t*(et_emp + np.sqrt(vt_emp)*np.array([-1,1])), color='black', linestyle='-',label='Empirical')
            ax.plot(units_k*(eg_emp + np.sqrt(vg_emp)*np.array([-1,1])), units_t*et_emp*np.ones(2), color='black', linestyle='-',label='Empirical')
            hdga, = ax.plot(units_k*eg_dga*np.ones(2),units_t*(et_dga + np.sqrt(vt_dga)*np.array([-1,1])), color='red', linestyle='-',label='DGA')
            ax.plot(units_k*(eg_dga + np.sqrt(vg_dga)*np.array([-1,1])), units_t*et_dga*np.ones(2), color='red', linestyle='-',label='DGA')
            ax.set_xlabel(r"$%s (%s)$"%(model.dam_dict[keys[k]]['name_full'],unit_symbol_k),fontdict=font)
            ax.set_ylabel(r"$%s (%s)$"%(model.dam_dict['one']['name_full'],unit_symbol_t),fontdict=font)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.set_title(r"Empirical correlations $B\to A$")
            ax.legend(handles=[hemp,hdga])
            fig.savefig(join(self.savefolder,"corr{}_ba".format(keys[k])))
            plt.close(fig)
            f.write("\t\tCorrelation with T: DGA: %3.3e, EMP: %3.3e\n"%(dga_corr,emp_corr))
            f.write("\t\tRate: DGA: %3.3e, EMP: %3.3e\n"%(dga_rate,emp_rate))
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
                f.write("\t\tMoment %d: DGA/time (t-weighted) = %3.3e, DGA/traj (t-weighted) = %3.3e,  DGA/time = %3.3e, DGA/traj = %3.3e, EMP/time (t-weighted) = %3.3e, EMP/traj (t-weighted) = %3.3e, EMP/time = %3.3e, EMP/traj = %3.3e\n" % (i,dga_avg_per_time_tweighted,dga_avg_per_traj_tweighted,dga_avg_per_time,dga_avg_per_traj,emp_avg_per_time_tweighted,emp_avg_per_traj_tweighted,emp_avg_per_time,emp_avg_per_traj))
                dga_moments_trajwise[1,i] = dga_avg_per_traj
                emp_moments_trajwise[1,i] = emp_avg_per_traj
            # Plot the moments for validation
            fig,ax = plt.subplots(ncols=2,figsize=(12,6),tight_layout=True,sharey=True)
            hdga, = ax[0].plot(np.arange(1,num_moments+1),dga_moments_trajwise[0,1:]**(1/np.arange(1,num_moments+1)),color='red',marker='o',linewidth=2,label='DGA')
            hemp, = ax[0].plot(np.arange(1,num_moments+1),emp_moments_trajwise[0,1:]**(1/np.arange(1,num_moments+1)),color='black',marker='o',linewidth=2,label='Empirical')
            ax[0].legend(handles=[hdga,hemp],prop={'size':18})
            ax[0].set_title(r"%s $(A\to B)$ moments"%model.dam_dict[keys[k]]['name'],fontdict=font)
            ax[0].set_xlabel("Moment number $k$",fontdict=font)
            ax[0].set_ylabel(r"$(E[(\int%s\,dt)^k])^{1/k}$"%model.dam_dict[keys[k]]['pay_symbol'],fontdict=font)
            #ax[0].set_yscale('log')
            ax[0].xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,num_moments+1)))
            ax[0].tick_params(axis='both',labelsize=14)
            hdga, = ax[1].plot(np.arange(1,num_moments+1),dga_moments_trajwise[1,1:]**(1/np.arange(1,num_moments+1)),color='red',marker='o',linewidth=2,label='DGA')
            hemp, = ax[1].plot(np.arange(1,num_moments+1),emp_moments_trajwise[1,1:]**(1/np.arange(1,num_moments+1)),color='black',marker='o',linewidth=2,label='Empirical')
            ax[1].legend(handles=[hdga,hemp],prop={'size':18})
            ax[1].set_title(r"%s $(B\to A)$ moments"%model.dam_dict[keys[k]]['name'],fontdict=font)
            ax[1].set_xlabel(r"Moment number $k$",fontdict=font)
            #ax[1].set_ylabel(r"$E[(\int_B^A%s\,dt)^n]$"%model.dam_dict[keys[k]]['pay_symbol'],fontdict=font)
            #ax[1].set_yscale('log')
            ax[1].xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,num_moments+1)))
            ax[1].tick_params(axis='both',labelsize=14)
            fig.savefig(join(self.savefolder,"moments_abba_log_{}".format(model.dam_dict[keys[k]]['abb_full'])))
            plt.close(fig)
            fig,ax = plt.subplots()# ncols=2,figsize=(12,6),tight_layout=True,sharey=True)
            hab, = ax.plot(np.arange(1,num_moments+1),dga_moments_trajwise[0,1:]/emp_moments_trajwise[0,1:],color='red',marker='o',linewidth=2,label=r"$\frac{DGA}{Emp.}\,(A\to B)$")
            hba, = ax.plot(np.arange(1,num_moments+1),dga_moments_trajwise[1,1:]/emp_moments_trajwise[1,1:],color='blue',marker='o',linewidth=2,label=r"$\frac{DGA}{Emp.}\,(B\to A)$")
            ax.plot(np.arange(1,num_moments+1),np.ones(num_moments),color='black',linestyle='--')
            ax.legend(handles=[hab,hba],prop={'size': 18})
            ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,num_moments+1)))
            ax.set_xlabel(r"Moment number $k$", fontdict=font)
            ax.tick_params(axis='both',labelsize=14)
            #ax[0].set_title("DGA/Empirical moment ratios",fontdict=font)
            ax.set_title(r"%s moment ratios"%model.dam_dict[keys[k]]["name"],fontsize='medium')
            fig.savefig(join(self.savefolder,"moments_abba_ratios_{}".format(model.dam_dict[keys[k]]['abb_full'])))
            plt.close(fig)
            # Also plot the full distribution and its approximation as Gamma
            fig,ax = plt.subplots(ncols=2,figsize=(12,6),tight_layout=True,sharey=True)
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
            hemp_pdf_ab, = ax[0].plot(bin_centers,hist,color='black',marker='o',label='Empirical PDF')
            hemp_gamma_ab, = ax[0].plot(bin_centers,emp_gamma_ab,color='blue',marker='o',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha_emp),helper.sci_fmt_latex(beta_emp)))
            hdga_gamma_ab, = ax[0].plot(bin_centers,dga_gamma_ab,color='red',marker='o',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha),helper.sci_fmt_latex(beta)))
            ax[0].set_title(r"$A\to B$ %s PDF"%model.dam_dict[keys[k]]["name"],fontdict=font)
            ax[0].legend(handles=[hemp_pdf_ab,hemp_gamma_ab,hdga_gamma_ab],prop={'size': 16})
            #ax[0].legend(handles=[hemp_pdf_ab,hdga_gamma_ab],prop={'size': 16})
            ax[0].tick_params(axis='both',labelsize=14)
            ax[0].set_xlabel("%s (%s)"%(model.dam_dict[keys[k]]["name"],model.dam_dict[keys[k]]["unit_symbol"]),fontdict=font)
            
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
            hemp_pdf_ba, = ax[1].plot(bin_centers,hist,color='black',marker='o',label='Empirical PDF')
            hemp_gamma_ba, = ax[1].plot(bin_centers,emp_gamma_ba,marker='o',color='blue',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha_emp),helper.sci_fmt_latex(beta_emp)))
            hdga_gamma_ba, = ax[1].plot(bin_centers,dga_gamma_ba,marker='o',color='red',label=r"$\Gamma(\alpha=%s,\beta=%s)$"%(helper.sci_fmt_latex(alpha),helper.sci_fmt_latex(beta)))
            ax[1].set_title(r"$B\to A$ %s PDF"%model.dam_dict[keys[k]]["name"],fontdict=font)
            ax[1].set_title(r"$B\to A$ PDF",fontdict=font)
            ax[1].legend(handles=[hemp_pdf_ba,hemp_gamma_ba,hdga_gamma_ba],prop={'size': 16})
            #ax[1].legend(handles=[hemp_pdf_ba,hdga_gamma_ba],prop={'size': 16})
            ax[1].tick_params(axis='both',labelsize=14)
            ax[1].set_xlabel("%s (%s)"%(model.dam_dict[keys[k]]["name"],model.dam_dict[keys[k]]["unit_symbol"]),fontdict=font)
            fig.savefig(join(self.savefolder,"pdf_{}".format(model.dam_dict[keys[k]]['abb_full'])))
            plt.close(fig)
        f.close()
        return
    def display_dam_moments_abba_current(self,model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs):
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
        num_obs = 5
        ab_obs_idx = np.random.choice(np.arange(len(ab_starts)),num_obs)
        ba_obs_idx = np.random.choice(np.arange(len(ba_starts)),num_obs)
        theta_ab_obs = []
        theta_ba_obs = []
        for i in range(num_obs):
            theta_ab_obs += [theta_2d_fun(x_long[ab_starts[ab_obs_idx[i]]:ab_ends[ab_obs_idx[i]]])]
            theta_ba_obs += [theta_2d_fun(x_long[ba_starts[ba_obs_idx[i]]:ba_ends[ba_obs_idx[i]]])]
        del x_long
        # If I substitute in F- and F+ for q- and q+, I guess we'll see where pathways accumulate the most of whatever damage function it's measuring
        Nx,Nt,xdim = data.X.shape
        keys = list(model.dam_dict.keys())
        num_moments = self.dam_moments[keys[0]]['xb'].shape[0]-1
        theta_xst = theta_2d_fun(model.tpt_obs_xst) # Possibly to be used as theta_ab
        for k in range(1):
            # -----------------------------
            # A->B
            fieldname = r"$A\to B$"  #r"$\pi_{AB},J_{AB}$"
            field = self.dam_moments[keys[k]]['ab'][0] # just (q-)*(q+)
            comm_bwd = self.dam_moments[keys[k]]['ax'][0]
            comm_fwd = self.dam_moments[keys[k]]['xb'][0]
            weight = self.chom
            theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
            xfw,tfw = model.load_least_action_path(self.physical_param_folder,dirn=1)
            theta_fw = theta_2d_fun(xfw)
            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=theta_fw,magu_obs=theta_ab_obs,cmap=plt.cm.YlOrBr,theta_ab=None,abpoints_flag=True)
            fig.savefig(join(self.savefolder,"piabj_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])))
            plt.close(fig)
            #sys.exit()
            # ---------------------------------
            # B->A
            fieldname = r"$B\to A$" #r"$\pi_{BA},J_{BA}$"
            field = self.dam_moments[keys[k]]['ba'][0] # just (q-)*(q+)
            comm_bwd = self.dam_moments[keys[k]]['bx'][0]
            comm_fwd = self.dam_moments[keys[k]]['xa'][0]
            weight = self.chom
            theta_x = theta_2d_fun(data.X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt,2))
            xfw,tfw = model.load_least_action_path(self.physical_param_folder,dirn=-1)
            theta_fw = theta_2d_fun(xfw)
            # Add current
            fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=theta_fw,magu_obs=theta_ba_obs,cmap=plt.cm.YlOrBr,theta_ab=None,abpoints_flag=True)
            fig.savefig(join(self.savefolder,"pibaj_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])))
            plt.close(fig)
        return
    def display_casts_abba(self,model,data,theta_2d_abbs):
        funlib = model.observable_function_library()
        Nx,Nt,xdim = data.X.shape
        keys = list(model.dam_dict.keys())
        num_moments = self.dam_moments[keys[0]]['xb'].shape[0]-1
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
            # Plot unconditional MFPT
            fig,ax = self.plot_field_2d(model,data,self.mfpt_b,weight,theta_x,shp=[20,20],fieldname=r"$E[\tau_B^+]$",fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
            fsuff = 'mfpt_xb_th0%s_th1%s'%(theta_2d_abbs[i][0],theta_2d_abbs[i][1])
            fig.savefig(join(self.savefolder,fsuff))
            plt.close(fig)
            fig,ax = self.plot_field_2d(model,data,self.mfpt_a,weight,theta_x,shp=[20,20],fieldname=r"$E[\tau_A^+]$",fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
            fsuff = 'mfpt_xa_th0%s_th1%s'%(theta_2d_abbs[i][0],theta_2d_abbs[i][1])
            fig.savefig(join(self.savefolder,fsuff))
            plt.close(fig)


            for k in range(len(keys)):
                print("\tStarting damage function %s"%(keys[k]))
                field_units = model.dam_dict[keys[k]]['units']
                # Determine vmin and vmax
                if keys[k] == 'vT':
                    vmin,vmax = 0,0.003
                elif keys[k] == 'one':
                    vmin,vmax = 0,250
                # Plot the actual function first
                field = field_units*model.dam_dict[keys[k]]['pay'](data.X[:,0]).reshape(-1,1)
                fieldname = model.dam_dict[keys[k]]['name']
                fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=None,comm_fwd=None,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                fsuff = '%s_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                fig.savefig(join(self.savefolder,fsuff))
                plt.close(fig)
                for j in range(min(3,self.num_moments)):
                    # A->B
                    comm_bwd = self.dam_moments[keys[k]]['ax'][0]
                    comm_fwd = self.dam_moments[keys[k]]['xb'][0]
                    if j == 0: 
                        fieldname = r"$P\{A\to B\}$"
                        field = field_units**j*self.dam_moments[keys[k]]['ab'][j] 
                    else:
                        prob = comm_bwd*comm_fwd
                        if j == 1: 
                            fieldname = r"$E[%s|A\to B]$"%(model.dam_dict[keys[k]]['name_full']) 
                            field = field_units**j*self.dam_moments[keys[k]]['ab'][j]
                        elif j == 2:
                            fieldname = r"$Var[%s|A\to B]$"%(model.dam_dict[keys[k]]['name_full']) 
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
                    fig.savefig(join(self.savefolder,fsuff))
                    plt.close(fig)
                    # ---------------------------------
                    # B->A
                    comm_bwd = self.dam_moments[keys[k]]['bx'][0]
                    comm_fwd = self.dam_moments[keys[k]]['xa'][0]
                    if j == 0: 
                        fieldname = r"$P\{B\to A\}$"
                        field = field_units**j*self.dam_moments[keys[k]]['ba'][j] 
                    else:
                        prob = comm_bwd*comm_fwd
                        if j == 1: 
                            fieldname = r"$E[%s|B\to A]$"%(model.dam_dict[keys[k]]['name_full']) 
                            field = field_units**j*self.dam_moments[keys[k]]['ba'][j]
                        elif j == 2:
                            fieldname = r"$Var[%s|B\to A]$"%(model.dam_dict[keys[k]]['name_full']) 
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
                    fig.savefig(join(self.savefolder,fsuff))
                    plt.close(fig)
                    # x->B
                    comm_fwd = self.dam_moments[keys[k]]['xb'][0]
                    comm_bwd = 1.0
                    if j == 0: 
                        fieldname = r"$P\{x\to B\}$"
                        field = field_units**j*self.dam_moments[keys[k]]['xb'][j] 
                    else:
                        prob = comm_bwd*comm_fwd
                        if j == 1: 
                            fieldname = r"$E[%s|x\to B]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                            field = field_units**j*self.dam_moments[keys[k]]['xb'][j]
                        elif j == 2:
                            fieldname = r"$Var[%s|x\to B]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                            field = field_units**j*(self.dam_moments[keys[k]]['xb'][j] - self.dam_moments[keys[k]]['xb'][1]**2)
                        field[np.where(prob==0)[0]] = np.nan
                        field *= 1.0/(prob + 1*(prob == 0))
                        print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                        # Correct now
                        field[np.where(field > vmax**j)[0]] = np.nan
                        field[np.where(field < vmin**j)[0]] = np.nan
                        print("field range: ({},{})".format(np.nanmin(field),np.nanmax(field)))
                    fig,ax = self.plot_field_2d(model,data,field,weight,theta_x,shp=[20,20],fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=True,current_flag=False,logscale=False,comm_bwd=comm_bwd,comm_fwd=comm_fwd,magu_fw=None,magu_obs=None,cmap=plt.cm.coolwarm,theta_ab=theta_xst,abpoints_flag=False,vmin=None,vmax=None)
                    fsuff = 'cast_%s%d_xb_th0%s_th1%s'%(model.dam_dict[keys[k]]['abb_full'],j,theta_2d_abbs[i][0],theta_2d_abbs[i][1])
                    fig.savefig(join(self.savefolder,fsuff))
                    plt.close(fig)
                    # ---------------------------------
                    # x->A
                    comm_fwd = self.dam_moments[keys[k]]['xa'][0]
                    comm_bwd = 1.0
                    if j == 0: 
                        fieldname = r"$P\{x\to A\}$"
                        field = field_units**j*self.dam_moments[keys[k]]['xa'][j] 
                    else:
                        prob = comm_bwd*comm_fwd
                        if j == 1: 
                            fieldname = r"$E[%s|x\to A]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                            field = field_units**j*self.dam_moments[keys[k]]['xa'][j]
                        elif j == 2:
                            fieldname = r"$Var[%s|x\to A]$"%(model.dam_dict[keys[k]]['name_fwd']) 
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
                    fig.savefig(join(self.savefolder,fsuff))
                    plt.close(fig)
                    # B->x
                    comm_fwd = self.dam_moments[keys[k]]['bx'][0]
                    comm_bwd = 1.0
                    if j == 0: 
                        fieldname = r"$P[B\to x]$"
                        field = field_units**j*self.dam_moments[keys[k]]['bx'][j] 
                    else:
                        prob = comm_bwd*comm_fwd
                        if j == 1: 
                            fieldname = r"$E[%s|B\to x]$"%(model.dam_dict[keys[k]]['name_bwd']) 
                            field = field_units**j*self.dam_moments[keys[k]]['bx'][j]
                        elif j == 2:
                            fieldname = r"$Var[%s|B\to x]$"%(model.dam_dict[keys[k]]['name_bwd']) 
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
                    fig.savefig(join(self.savefolder,fsuff))
                    plt.close(fig)
                    # ---------------------------------
                    # A->x
                    comm_fwd = self.dam_moments[keys[k]]['ax'][0]
                    comm_bwd = 1.0
                    if j == 0: 
                        fieldname = r"$P[A\to x]$"
                        field = field_units**j*self.dam_moments[keys[k]]['ax'][j] 
                    else:
                        prob = comm_bwd*comm_fwd
                        if j == 1: 
                            fieldname = r"$E[%s|A\to x]$"%(model.dam_dict[keys[k]]['name_fwd']) 
                            field = field_units**j*self.dam_moments[keys[k]]['ax'][j]
                        elif j == 2:
                            fieldname = r"$Var[%s|A\to x]$"%(model.dam_dict[keys[k]]['name_fwd']) 
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
                    fig.savefig(join(self.savefolder,fsuff))
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
        theta2d_test_idx_lower[0,0] = nnidx[np.argmin(np.abs(qth_qth1half[nnidx]-1/3))]
        theta2d_test_idx_lower[1,0] = nnidx[np.argmin(np.abs(qth_qth1half[nnidx]-2/3))]
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
        # -------------------------------------------
        # Plots
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
        ss0 = ss0[np.linspace(0,len(ss0)-1,num_series).astype(int)]
        ss1 = np.sort(long_test_idx_1[np.where(np.isnan(self.dam_emp['one']['x_Dc'][long_test_idx_1])==0)[0]])
        ss1 = ss1[np.linspace(0,len(ss1)-1,num_series).astype(int)]
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
            htj_emp, = ax[j].plot(tb_emp[j]*np.ones(2),theta2d_units[1]*theta_ab[:,1],color='black',linestyle='--',linewidth=4,label=r"$\overline{\tau_B}(\theta_{%d})=%.1f$"%(j,tb_emp[j]))
            dthab = np.abs(theta_ab[0,1]-theta_ab[1,1])
            ax[j].text(0,theta2d_units[1]*(theta_ab[0,1]+0.01*dthab),asymb,fontdict=bigfont,color='black',weight='bold')
            ax[j].text(0,theta2d_units[1]*(theta_ab[1,1]+0.01*dthab),bsymb,fontdict=bigfont,color='black',weight='bold')
            ax[j].tick_params(axis='both',labelsize=25)
            ax[j].legend(handles=handles[j]+[htj_dga,htj_emp],prop={'size':25})
            ax[j].text(max_length*4/8,theta2d_units[1]*(theta_ab[0,1]*0.5+theta_ab[1,1]*0.5),r"$E[q^+|\theta_{%d}]=%.2f;\ \frac{N_B}{N}(\theta_{%d})=%.2f$"
            "\n"
            r"$E[\tau^+|\theta_{%d}\to B]=%.1f;\ \overline{T_B}(\theta_{%d})=%.1f$"
            %(j,test_qth[j],j,Nb[j]/num_series,j,test_tbth[j],j,tb_emp[j]),fontdict=font)
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
            ax.set_xlabel(r"$q^+$ Empirical", fontdict=font)
            ax.set_ylabel(r"$q^+$ DGA", fontdict=font)
            ax.set_title(r"$N=%s,\ M=%d$, Lag$=$%d days"%(helper.sci_fmt_latex0(self.nshort),basis_size,self.lag_time_seq[-1]),fontdict=font)
            fig.savefig(join(self.savefolder,"fidelity_qp"),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)
            # ---------------------------------------------
            # One: Plot them both as a function of U (30 km)
            fig,ax = plt.subplots(figsize=(6,6))
            hemp, = ax.plot(theta_1d_units*thaxes[0],q_long_grid,color='black',marker='o',label=r"$q^+_{EMP}$")
            hdga, = ax.plot(theta_1d_units*thaxes[0],q_short_grid,color='red',marker='o',label=r"$q^+_{DGA}$")
            herr, = ax.plot([],[],color='white',label=r"RMS error $\epsilon=%.3f$"%(total_error))
            #ax.plot(q_long_grid+np.sqrt(q_var_long_grid/N_long_grid),q_long_grid,linestyle='--',color='black')
            #ax.plot(q_long_grid-np.sqrt(q_var_long_grid/N_long_grid),q_long_grid,linestyle='--',color='black')
            ax.legend(handles=[hemp,hdga,herr],prop={'size': 14},loc='lower left')
            ax.set_xlabel(r"%s (%s)"%(theta_1d_name,theta_1d_unit_symbol), fontdict=font)
            ax.set_ylabel(r"$q^+$ DGA, Empirical", fontdict=font)
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
                    uname = r"$P[A\to x]$"
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
                    uname = r"$P[B\to x]$"
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
    def display_1d_densities(self,model,data,theta_1d_abbs,theta_1d_orientations,fig=None,ax=None,include_reactive=True):
        funlib = model.observable_function_library()
        keys = list(model.dam_dict.keys())
        field = self.dam_moments[keys[0]]['ab'][0] # just (q-)*(q+)
        comm_bwd = self.dam_moments[keys[0]]['ax'][0,:,0]
        comm_fwd = self.dam_moments[keys[0]]['xb'][0,:,0]
        piab = comm_fwd*comm_bwd
        piab *= 1.0/np.sum(piab*self.chom)
        piba = (1-comm_fwd)*(1-comm_bwd)
        piba *= 1.0/np.sum(piba*self.chom)
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
            if include_reactive:
                _,_,hpiab = helper.plot_field_1d(theta_x,piab,weight,avg_flag=False,color='darkorange',label=r"$\pi_{AB}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                _,_,hpiba = helper.plot_field_1d(theta_x,piba,weight,avg_flag=False,color='springgreen',label=r"$\pi_{BA}$",orientation=theta_1d_orientations[k],unit_symbol=theta_1d_unit_symbol,units=theta_1d_units,fig=fig,ax=ax,thetaname=theta_1d_name,density_flag=True,linewidth=2.5)
                handles += [hpiab,hpiba]
            #ax.legend(handles=handles,prop={'size': 25})
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
            self.display_dam_moments_abba_current(model,data,theta_2d_fun,theta_2d_names,theta_2d_units,theta_2d_unit_symbols,theta_2d_abbs[k])
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
        fieldname = r"$\pi,J$"
        weight = self.chom
        fig,ax = self.plot_field_2d(model,data,field,weight,theta_2d_short,fieldname=fieldname,fun0name=theta_2d_names[0],fun1name=theta_2d_names[1],units=theta_2d_units,unit_symbols=theta_2d_unit_symbols,avg_flag=False,current_flag=True,logscale=True,comm_bwd=comm_bwd,comm_fwd=comm_fwd,cmap=plt.cm.YlOrBr)
        fig.set_tight_layout(True)
        fig.savefig(join(self.savefolder,"pij_{}_{}".format(theta_2d_abbs[0],theta_2d_abbs[1])))
        plt.close(fig)
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
    def project_current_new(self,model,data,theta_x,comm_bwd,comm_fwd):
        # compute J_(AB)\cdot\nabla\theta. theta is a multi-dimensional observable, so we end up with a vector of that size.
        # This should be used hopefully for maximizing the reactive flux on a surface. 
        Nx,Nt,thdim = theta_x.shape
        Jtheta = np.zeros((Nx,thdim))
        bdy_dist = lambda x: np.minimum(model.adist(x),model.bdist(x))
        data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current)
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
        field0 = (chomss*comm_bwd_x*comm_fwd_ypj*(theta_ypj - theta_x).T).T
        field1 = (chomss*comm_bwd_xpj*comm_fwd_yj*(theta_yj - theta_xpj).T).T
        print("field0.shape = {}, field1.shape = {}".format(field0.shape,field1.shape))
        #sys.exit()
        #    if dirn == -1:
        #        comm_bwd = 1 - comm_bwd
        #        comm_bwdpj = 1 - comm_bwdpj
        #        comm_fwd_yj = 1 - comm_fwd_yj
        #        comm_fwd_ypj = 1 - comm_fwd_ypj
        #    field0 = (chomss*comm_bwd*comm_fwd_ypj*(theta_ypj - theta_x).T).T
        #    field1 = (chomss*comm_bwdpj*comm_fwd_yj*(theta_yj - theta_xpj).T).T
        #else:
        #    field0 = (chomss*(theta_yj - theta_x).T).T
        #    field1 = field0
        thdim = field0.shape[1]
        if shp is None: shp = 20*np.ones(thdim,dtype=int) # number of INTERIOR
        Ncell = np.prod(shp)
        J = np.zeros((Ncell,thdim))
        for d in range(thdim):
            _,dth,thaxes,cgrid,J0,J0_std,_,_,_ = helper.project_field(field0[:,d],chomss,theta_x,shp=shp,avg_flag=False)
            _,dth,thaxes,cgrid,J1,J1_std,_,_,_ = helper.project_field(field1[:,d],chomss,theta_yj,shp=shp,avg_flag=False)
            J[:,d] = 1/(2*self.lag_time_current_display*np.prod(dth))*(J0 + J1)
        return thaxes,J
    def plot_field_2d(self,model,data,field,weight,theta_x,shp=[60,60],cmap=plt.cm.coolwarm,fieldname="",fun0name="",fun1name="",current_flag=False,comm_bwd=None,comm_fwd=None,current_shp=[25,25],abpoints_flag=False,theta_ab=None,avg_flag=True,logscale=False,ss=None,magu_fw=None,magu_obs=None,units=np.ones(2),unit_symbols=["",""],cbar_orientation='horizontal',fig=None,ax=None,vmin=None,vmax=None):
        # theta_x is the observable on all of the x's
        # First plot the scalar
        #print("About to call Helper to plot 2d")
        fig,ax = helper.plot_field_2d(field[:,0],weight,theta_x[:,0],shp=shp,cmap=cmap,fieldname=fieldname,fun0name=fun0name,fun1name=fun1name,avg_flag=avg_flag,std_flag=False,logscale=logscale,ss=ss,units=units,unit_symbols=unit_symbols,cbar_orientation=cbar_orientation,fig=fig,ax=ax,vmin=vmin,vmax=vmax)
        ax.set_xlim([np.min(theta_x[:,:,0])*units[0],np.max(theta_x[:,:,0])*units[0]])
        ax.set_ylim([np.min(theta_x[:,:,1])*units[1],np.max(theta_x[:,:,1])*units[1]])
        #print("Helper's role is done")
        if abpoints_flag:
            #ass = np.where(np.in1d(ss,self.aidx))[0]
            #ass_theta = np.random.choice(ass,min(2000,len(ass)),replace=False)
            ass_theta = self.aidx
            #bss = np.where(np.in1d(ss,self.bidx))[0]
            #bss_theta = np.random.choice(bss,min(2000,len(bss)),replace=False)
            bss_theta = self.bidx
            ax.scatter(units[0]*theta_x[ass_theta,0,0],units[1]*theta_x[ass_theta,0,1],color='lightgray',marker='.',zorder=2)
            ax.scatter(units[0]*theta_x[bss_theta,0,0],units[1]*theta_x[bss_theta,0,1],color='lightgray',marker='.',zorder=2)
            print("ABpoints done")
        if theta_ab is not None:
            ax.text(units[0]*theta_ab[0,0],units[1]*theta_ab[0,1],asymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=15,horizontalalignment='center',verticalalignment='center',zorder=100)
            ax.text(units[0]*theta_ab[1,0],units[1]*theta_ab[1,1],bsymb,bbox=dict(facecolor='white',alpha=1.0),color='black',fontsize=15,horizontalalignment='center',verticalalignment='center',zorder=100)
        ax.set_title("{}".format(fieldname),fontdict=font)
        xlab = fun0name
        if len(unit_symbols[0]) > 0: xlab += " ({})".format(unit_symbols[0])
        ylab = fun1name
        if len(unit_symbols[1]) > 0: ylab += " ({})".format(unit_symbols[1])
        ax.set_xlabel("{}".format(xlab),fontdict=font)
        ax.set_ylabel("{}".format(ylab),fontdict=font)
        if current_flag:
            bdy_dist = lambda x: np.minimum(model.adist(x),model.bdist(x))
            data.insert_boundaries(bdy_dist,lag_time_max=self.lag_time_current_display)
            #theta_yj,theta_xpj,theta_ypj = thetaj # thetaj better not be None either
            print("comm_bwd.shape = {}".format(comm_bwd.shape))
            print("comm_fwd.shape = {}".format(comm_fwd.shape))
            print("About to go into plotting current")
            thaxes_current,J = self.project_current(data,theta_x[:,0],theta_x[np.arange(data.nshort),data.last_idx],theta_x[np.arange(data.nshort),data.last_entry_idx],theta_x[np.arange(data.nshort),data.first_exit_idx],current_shp,comm_bwd,comm_fwd,ss=ss)
            dth = np.array([thax[1] - thax[0] for thax in thaxes_current])
            Jmag_full = np.sqrt(np.sum(J**2, 1))
            minmag,maxmag = np.nanmin(Jmag_full),np.nanmax(Jmag_full)
            #print("minmag={}, maxmag={}".format(minmag,maxmag))
            dsmin,dsmax = np.max(current_shp)/40,np.max(current_shp)/5 # lengths if arrows in grid box units
            th0_subset = np.arange(current_shp[0]) #np.linspace(0,p[0]-2,current_shp[0]-2)).astype(int)
            th1_subset = np.arange(current_shp[1]) #np.linspace(0,len(thaxes_current[1])-2,min(shp[1]-2,current_shp[1]-2)).astype(int)
            J0 = J[:,0].reshape(current_shp)[th0_subset,:][:,th1_subset] #*units[0]
            J1 = J[:,1].reshape(current_shp)[th0_subset,:][:,th1_subset] #*units[1]
            Jmag = np.sqrt(J0**2 + J1**2)
            print("Jmag range = ({},{})".format(np.nanmin(Jmag),np.nanmax(Jmag)))
            print("J0.shape = {}".format(J0.shape))
            ds = dsmin + (dsmax - dsmin)*(Jmag - minmag)/(maxmag - minmag)
            normalizer = ds*(Jmag != 0)/(np.sqrt((J0/(dth[0]))**2 + (J1/(dth[1]))**2) + (Jmag == 0))
            J0 *= normalizer*(1 - np.isnan(J0))
            J1 *= normalizer*(1 - np.isnan(J1))
            print("Final J0 range = ({},{})".format(np.nanmin(J0),np.nanmax(J0)))
            th01_subset,th10_subset = np.meshgrid(units[0]*thaxes_current[0][th0_subset],units[1]*thaxes_current[1][th1_subset],indexing='ij')
            ax.quiver(th01_subset,th10_subset,units[0]*J0,units[1]*J1,angles='xy',scale_units='xy',scale=1.0,color='black',width=1.4,headwidth=4.0,units='dots',zorder=4) # was width=2.0, headwidth=2.7
        if magu_obs is not None:
            for ti in range(len(magu_obs)):
                ax.plot(magu_obs[ti][:,0]*units[0],magu_obs[ti][:,1]*units[1],color='deepskyblue',zorder=3,alpha=1.0,linestyle='solid',linewidth=0.85)
        if magu_fw is not None:
            ax.plot(magu_fw[:,0]*units[0],magu_fw[:,1]*units[1],color='cyan',linewidth=2.0,zorder=5,linestyle='solid')
        fig.set_tight_layout(True)
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
    def out_of_sample_extension(self,field,data,xnew):
        # For a new sample (such as a long trajectory), extend the field to the new ones just by nearest-neighbor averaging
        k = 15
        np.random.seed(0)
        good_idx = np.where(np.isnan(field)==0)[0]
        #ss = np.random.choice(np.arange(self.nshort),size=min(self.nshort,100000),replace=False)
        ss = np.random.choice(good_idx,size=min(len(good_idx),100000),replace=False)
        Xsq = np.sum(data.X[ss,0]**2,1)
        dsq = np.add.outer(Xsq,np.sum(xnew**2,1)) - 2*data.X[ss,0].dot(xnew.T)
        knn = np.argpartition(dsq,k+1,axis=0)
        knn = knn[:k,:]
        close_dsq = np.zeros((k,len(xnew)))
        for j in range(k):
            close_dsq[j] = dsq[knn[j],np.arange(len(xnew))]
        dsq = dsq[knn,np.arange(len(xnew))]
        close_field = field[ss[knn]]
        weights = np.exp(-0*close_dsq)
        weights = weights/np.sum(weights*(np.isnan(close_field)==0),0)
        fnew = np.nansum(close_field*weights,0)
        frange = np.nanmax(field) - np.nanmin(field)
        if np.nanmax(fnew) > np.nanmax(field) + 0.05*frange:
            sys.exit("ERROR: in out-of-sample-extension, nanmax(fnew) = {} while nanmax(field) = {}".format(np.nanmax(fnew),np.nanmax(field)))
        if np.nanmin(fnew) < np.nanmin(field) - 0.05*frange:
            sys.exit("ERROR: in out-of-sample-extension, nanmin(fnew) = {} while nanmin(field) = {}".format(np.nanmin(fnew),np.nanmin(field)))

        #fnew = np.zeros(len(xnew))
        #for i in range(len(xnew)):
        #    dsq = Xsq + np.sum(xnew[i]**2) - 2*self.X0[ss].dot(xnew[i])
        #    knn = np.argpartition(dsq,k+1)[:k]
        #    weights = np.exp(-0*dsq[knn])
        #    fnew[i] = np.sum(field[ss[knn]]*weights)/np.sum(weights)
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
    def plot_prediction_curves(self):
        # VESTIGIAL
        # Plot committor vs. lead time
        # TODO: swap in modern names for mfpt, and also do the distribution thing
        eps = 1e-2
        # Define all the times and densities
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
        ymax = np.nanquantile(np.concatenate((tb,ta)),0.95)
        print("ymin = {}, ymax = {}".format(ymin,ymax))
        print("A avoiding B in range {},{}".format(np.nanmin(ta),np.nanmax(ta)))
        # x -> B
        ss_xb = np.random.choice(np.arange(self.nshort),size=min(self.nshort,30000),replace=True,p=piab)
        fig,ax = plt.subplots()
        ax.scatter(qb[ss_xb],tb[ss_xb],color='black',marker='.',s=10,alpha=1.0,zorder=-1)
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel(r"$P\{x\to B\}$",fontdict=font)
        ax.set_ylabel(r"$E[\tau_B|x\to B]$ (days)",fontdict=font)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.tick_params(axis='both', which='major', labelsize=15)
        #fig,ax = self.plot_field_2d(piab,np.ones(self.nshort),theta_x,shp=[25,25],cmap=plt.cm.summer,fieldname="",fun0name=r"Forward committor",fun1name=r"Time to $B$",current_flag=False,thetaj=None,abpoints_flag=False,std_flag=False,logscale=True)
        coeffs = self.weighted_least_squares(qb,tb,self.chom)
        print("prediction curves coeffs = {}".format(coeffs))
        x = np.array([np.min(qb),np.max(qb)])
        symbol = "+" if coeffs[1]>0 else ""
        handle, = ax.plot(x,coeffs[0]+coeffs[1]*x,color='black',linewidth=3,label=r"$%.1f%s%.1fP\{x\to B\}$" % (coeffs[0],symbol,coeffs[1]))
        ax.legend(handles=[handle],prop={'size': 20})
        ax.set_title(r"Breakdown",fontdict=font)
        ax.tick_params(axis='both',which='major',labelsize=15)
        fig.savefig(join(self.savefolder,"pi_Tb_comm"))
        plt.close(fig)
        # x -> A
        ss_xa = np.random.choice(np.arange(self.nshort),size=min(self.nshort,30000),replace=True,p=piba)
        fig,ax = plt.subplots()
        ta[np.where(qa < eps)[0]] = np.nan
        ax.scatter(qa[ss_xa],ta[ss_xa],color='black',marker='.',s=10,alpha=1.0,zorder=-1)
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel(r"$P\{x\to A\}$",fontdict=font)
        ax.set_ylabel(r"$E[\tau_A|x\to A]$ (days)",fontdict=font)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.tick_params(axis='both', which='major', labelsize=15)
        coeffs = self.weighted_least_squares(qa,ta,self.chom)
        print("prediction curve coeffs = {}".format(coeffs))
        x = np.array([0,1])
        symbol = "+" if coeffs[1]>0 else ""
        handle, = ax.plot(x,coeffs[0]+coeffs[1]*x,color='black',linewidth=3,label=r"$%0.1f%s%0.1fP\{x\to A\}$" % (coeffs[0],symbol,coeffs[1]))
        ax.legend(handles=[handle],prop={'size': 20})
        ax.set_title(r"Recovery",fontdict=font)
        fig.savefig(join(self.savefolder,"pi_Ta_1-comm"))
        plt.close(fig)
        print("Prediction curves plotted")
        return
    def maximize_rflux_on_surface(self,model,data,theta_x,comm_bwd,comm_fwd,weight,theta_level,theta_tol,max_num_states):
        print("theta_x.shape = {}".format(theta_x.shape))
        Jup,Jdn = self.project_current_new(model,data,theta_x,comm_bwd,comm_fwd)
        print("Jup.shape = {}, Jdn.shape = {}".format(Jup.shape,Jdn.shape))
        print("theta_level = {}".format(theta_level))
        Jup = Jup.flatten()
        Jdn = Jdn.flatten()
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current/2))
        idx = np.where(np.sqrt(np.sum((theta_x[:,tidx] - theta_level)**2, 1)) < theta_tol)[0]
        print("\tAt level {}, len(idx) = {}".format(theta_level,len(idx)))
        if len(idx) == 0:
            idx = [np.argmin(np.abs(theta_x - theta_level))]
            print("WARNING: no datapoints are close to the level")
        # Maximize reactive density constrained to the surface
        rflux = np.abs(Jdn[idx] + Jup[idx])
        num = min(max_num_states,len(idx))
        rflux_max_idx = np.argpartition(-rflux,num)[:num]
        return idx[rflux_max_idx],rflux[rflux_max_idx],theta_x[idx[rflux_max_idx],tidx]
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
    def plot_transition_states(self,model,data):
        num_per_level = 3
        # Plot dominant transition states
        #funlib = hm.observable_function_library(q)
        Nx,Nt,xdim = data.X.shape
        key = list(self.dam_moments.keys())[0]
        comm_fwd = self.dam_moments[key]['xb'][0,:,:]
        comm_bwd = self.dam_moments[key]['ax'][0,:,:]
        weight = np.ones(data.nshort)/data.nshort #self.chom
        qlevels = np.array([0.25,0.5,0.75])
        # A -> B
        reac_dens_idx = np.zeros((len(qlevels),num_per_level),dtype=int)
        real_qlevels = np.zeros((len(qlevels),num_per_level))
        colorlist = []
        for i in range(len(qlevels)):
            reac_dens_idx[i,:],reac_dens_weights,ans2 = self.maximize_rflux_on_surface(model,data,comm_fwd.reshape((Nx,Nt,1)),comm_bwd,comm_fwd,weight,qlevels[i],0.05,num_per_level)
            print("ans2.shape = {}".format(ans2.shape))
            real_qlevels[i,:] = qlevels[i] 
            color = 'black' if i == 1 else plt.cm.coolwarm(qlevels[i]) 
            colorlist += [color for j in range(num_per_level)]
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current/2))
        fig,ax = model.plot_multiple_states(data.X[reac_dens_idx.flatten(),tidx],real_qlevels.flatten(),r"q^+",colorlist=colorlist)
        ax.set_title(r"$A\to B$ transition states",fontdict=font)
        fig.savefig(join(self.savefolder,"trans_states_ab"))
        plt.close(fig)
        # B -> A
        reac_dens_idx = np.zeros((len(qlevels),num_per_level),dtype=int)
        real_qlevels = np.zeros((len(qlevels),num_per_level))
        for i in range(len(qlevels)):
            reac_dens_idx[i,:],reac_dens_weights,ans2 = self.maximize_rflux_on_surface(model,data,comm_fwd.reshape((Nx,Nt,1)),1-comm_bwd,1-comm_fwd,weight,qlevels[i],0.05,num_per_level)
            real_qlevels[i,:] = qlevels[i] 
        tidx = np.argmin(np.abs(data.t_x - self.lag_time_current/2))
        fig,ax = model.plot_multiple_states(data.X[reac_dens_idx.flatten(),tidx],real_qlevels.flatten(),r"q^+")
        ax.set_title(r"$B\to A$ transition states",fontdict=font)
        fig.savefig(join(self.savefolder,"trans_states_ba"))
        plt.close(fig)
        return

