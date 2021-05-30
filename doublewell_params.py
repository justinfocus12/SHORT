# Doublewell parameters
import numpy as np

def get_algo_params():
    # Algorithm parameters
    tmax_long = 50000.0
    tmax_short = 0.75
    dt_save = 0.001
    dt_sim = 0.001
    nshort = 500000 #200000
    basis_type = 'MSM'
    basis_size = 300 
    lag_time = 0.2 
    nlags = 21 
    lag_time_seq = np.linspace(0,lag_time,nlags)
    lag_time_current = lag_time_seq[1] # For computing rates
    lag_time_current_display = lag_time_seq[4] # For displaying current vector fields
    min_clust_size = 5
    if min_clust_size*basis_size > nshort:
        sys.exit("ERROR: basis_size*min_clust_size > nshort")
    # Return a dictionary
    algo_params = dict({
        'tmax_long': tmax_long,
        'tmax_short': tmax_short,
        'dt_save': dt_save,
        'dt_sim': dt_sim,
        'nshort': nshort,
        'basis_type': basis_type,
        'basis_size': basis_size,
        'lag_time': lag_time,
        'lag_time_current': lag_time_current,
        'lag_time_current_display': lag_time_current_display,
        'min_clust_size': min_clust_size,
        })
    # Make a string for the corresponding folder
    algo_param_string = ("tlong{}_N{}_bs{}_lag{}_nlags{}_lagj{}".format(tmax_long,nshort,basis_size,lag_time,nlags,lag_time_current)).replace('.','p')
    return algo_params,algo_param_string

def get_physical_params():
    tau = 0.25
    kappa = 0.0
    sigma = 1.0
    state_dim = 2
    obs_dim = 2 # Observable degrees of freedom
    physical_params = dict({
        'tau': tau,
        'kappa': kappa,
        'sigma': sigma,
        'state_dim': state_dim,
        'obs_dim': obs_dim,
        })
    physical_param_string = ("tau{}_kappa{}_sigma{}".format(tau,kappa,sigma)).replace(".","p")
    return physical_params,physical_param_string


