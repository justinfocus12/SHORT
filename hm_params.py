# Doublewell parameters
import numpy as np

def get_algo_params():
    # Algorithm parameters
    tmax_long = 1000000.0
    tmax_short = 20.0
    dt_save = 0.5
    nshort = 500000 
    basis_type = 'MSM'
    basis_size = 1500 
    lag_time = 20.0 
    nlags = 21 
    lag_time_seq = np.linspace(0,lag_time,nlags)
    lag_time_current = lag_time_seq[1] # For computing rates
    lag_time_current_display = lag_time_seq[4] # For displaying current vector fields
    max_clust_per_level = 100
    min_clust_size = 10
    if min_clust_size*basis_size > nshort:
        sys.exit("ERROR: basis_size*min_clust_size > nshort")
    # Return a dictionary
    algo_params = dict({
        'tmax_long': tmax_long,
        'tmax_short': tmax_short,
        'dt_save': dt_save,
        'nshort': nshort,
        'basis_type': basis_type,
        'basis_size': basis_size,
        'lag_time': lag_time,
        'lag_time_seq': lag_time_seq,
        'lag_time_current': lag_time_current,
        'lag_time_current_display': lag_time_current_display,
        'min_clust_size': min_clust_size,
        'max_clust_per_level': max_clust_per_level,
        })
    # Make a string for the corresponding folder
    algo_param_string = ("tlong{}_N{}_bs{}_lag{}_nlags{}_lagj{}".format(tmax_long,nshort,basis_size,lag_time,nlags,lag_time_current)).replace('.','p')
    return algo_params,algo_param_string

def get_physical_params():
    du_per_day = 1.0
    hB_d = 38.5
    abdefdim =               1
    ref_alt =                30.0 # 21.5 or 26.9 or 29.6
    dt_sim = 0.005
    physical_params = dict({
        'hB_d': hB_d,
        'du_per_day': du_per_day,
        'dt_sim': dt_sim,
        'ref_alt': ref_alt,
        'abdefdim': abdefdim,
        })
    physical_param_string = ("du{}_h{}_abdd{}".format(du_per_day,hB_d,abdefdim)).replace(".","p")
    return physical_params,physical_param_string


