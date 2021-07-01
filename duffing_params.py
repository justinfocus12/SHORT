# Doublewell parameters
import numpy as np

def get_algo_params():
    # Algorithm parameters
    tmax_long = 50000.0
    tmax_short = 10.0
    dt_save = 0.005
    nshort = 500000 
    basis_type = 'MSM'
    basis_size = 1000
    lag_time = 10.0
    nlags = 11 
    num_moments = 2
    lag_time_seq = np.linspace(0,lag_time,nlags)
    lag_time_current = lag_time_seq[1] # For computing rates
    lag_time_current_display = lag_time_seq[4] # For displaying current vector fields
    max_clust_per_level = 100
    min_clust_size = 10
    if min_clust_size*basis_size > nshort:
        sys.exit("ERROR: basis_size*min_clust_size > nshort")
    sampling_feature_names = ['x','v']
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
        })
    # Make a string for the corresponding folder
    algo_param_string = ("tlong{}_N{}_bs{}_lag{}_nlags{}_lagj{}_sf0{}".format(tmax_long,nshort,basis_size,lag_time,nlags,lag_time_current,sampling_feature_names[0])).replace('.','p')
    return algo_params,algo_param_string

def get_physical_params():
    sigma_x = 0.0
    sigma_p = 0.2
    dt_sim = 0.005
    kappa = 0.1
    alpha = 1.0
    mu = 10.0
    physical_params = dict({
            'sigma_x': sigma_x,
            'sigma_p': sigma_p,
            'dt_sim': dt_sim,
            'kappa': kappa,
            'alpha': alpha,
            'mu': mu,
        })
    physical_param_string = ("k{}_a{}_m{}_sx{}_sp{}".format(kappa,alpha,mu,sigma_x,sigma_p)).replace(".","p")
    return physical_params,physical_param_string


