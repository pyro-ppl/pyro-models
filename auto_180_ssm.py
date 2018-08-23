# model file: ../example-models/BPA/Ch.05/ssm.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    T = data["T"]
    y = data["y"]
    check_constraints(T, low=0, dims=[1])
    check_constraints(y, dims=[T])

def init_params(data, params):
    # initialize data
    T = data["T"]
    y = data["y"]
    # assign init values for parameters
    params["mean_lambda"] = init_real("mean_lambda", low=0, high=10) # real/double
    params["sigma_proc"] = init_real("sigma_proc", low=0, high=10) # real/double
    params["sigma_obs"] = init_real("sigma_obs", low=0, high=100) # real/double
    params["lambda_"] = init_vector("lambda_", low=0, dims=((T - 1))) # vector
    params["N_est1"] = init_real("N_est1", low=0, high=500) # real/double

def model(data, params):
    # initialize data
    T = data["T"]
    y = data["y"]
    # INIT parameters
    mean_lambda = params["mean_lambda"]
    sigma_proc = params["sigma_proc"]
    sigma_obs = params["sigma_obs"]
    lambda_ = params["lambda_"]
    N_est1 = params["N_est1"]
    # initialize transformed parameters
    N_est = init_vector("N_est", low=0, dims=(T)) # vector
    N_est[1 - 1] = _pyro_assign(N_est[1 - 1], N_est1)
    for t in range(1, to_int((T - 1)) + 1):
        N_est[(t + 1) - 1] = _pyro_assign(N_est[(t + 1) - 1], (_index_select(N_est, t - 1)  * _index_select(lambda_, t - 1) ))
    # model block

    lambda_ =  _pyro_sample(lambda_, "lambda_", "normal", [mean_lambda, sigma_proc])
    y =  _pyro_sample(y, "y", "normal", [N_est, sigma_obs], obs=y)

