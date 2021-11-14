# Doublewell parameters
import numpy as np

def get_algo_params():
    # Algorithm parameters
    tmax_long = 200000.0
    tmax_short = 20.0
    dt_save = 0.5
    nshort = 200000
    istart = 0 # Highly dependent on the existing static database
    basis_type = 'MSM'
    basis_size = 1000 
    lag_time = 10.0 
    nlags = int(lag_time/dt_save) + 1 
    num_moments = 2
    lag_time_seq = np.linspace(0,lag_time,nlags)
    lag_time_current = lag_time_seq[1] # For computing rates
    lag_time_current_display = lag_time_seq[4] # For displaying current vector fields
    max_clust_per_level = 100
    min_clust_size = 10
    if min_clust_size*basis_size > nshort:
        sys.exit("ERROR: basis_size*min_clust_size > nshort")
    sampling_feature_names = ['magref','Uref']
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
        'sampling_feature_names': sampling_feature_names,
        'sampling_feature_suffix': "_sf0{}".format(sampling_feature_names[0]),
        'num_moments': num_moments,
        'istart': istart,
        })
    # Make a string for the corresponding folder
    algo_param_string = ("istart{}_tlong{}_N{}_bs{}_lag{}_nlags{}_lagj{}_sf0{}".format(istart,tmax_long,nshort,basis_size,lag_time,nlags,lag_time_current,sampling_feature_names[0])).replace('.','p')
    return algo_params,algo_param_string

def get_physical_params():
    du_per_day = 1.0
    hB_d = 38.5
    length = 2.5e5
    time = 86500.0 
    abdefdim =               1
    radius_a =               0.0
    radius_b =               10.0/(length/time)
    ref_alt =                30.0 # 21.5 or 26.9 or 29.6
    dt_sim = 0.005
    physical_params = dict({
        'hB_d': hB_d,
        'du_per_day': du_per_day,
        'dt_sim': dt_sim,
        'ref_alt': ref_alt,
        'abdefdim': abdefdim,
        'radius_a': radius_a,
        'radius_b': radius_b,
        })
    physical_param_string = ("du{}_h{}_abdd{}_rada{}_radb{}".format(du_per_day,hB_d,abdefdim,radius_a,radius_b)).replace(".","p")
    return physical_params,physical_param_string


