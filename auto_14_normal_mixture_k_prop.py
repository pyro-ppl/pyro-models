# model file: ../example-models/basic_estimators/normal_mixture_k_prop.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    K = data["K"]
    N = data["N"]
    y = data["y"]
    check_constraints(K, low=1, dims=[1])
    check_constraints(N, low=1, dims=[1])
    check_constraints(y, dims=[N])

def init_params(data, params):
    # initialize data
    K = data["K"]
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["theta"] = init_simplex("theta") # real/double
    params["mu_prop"] = init_simplex("mu_prop") # real/double
    params["mu_loc"] = init_real("mu_loc") # real/double
    params["mu_scale"] = init_real("mu_scale", low=0) # real/double
    params["sigma"] = init_real("sigma", low=0, dims=(K)) # real/double

def model(data, params):
    # initialize data
    K = data["K"]
    N = data["N"]
    y = data["y"]
    # INIT parameters
    theta = params["theta"]
    mu_prop = params["mu_prop"]
    mu_loc = params["mu_loc"]
    mu_scale = params["mu_scale"]
    sigma = params["sigma"]
    # initialize transformed parameters
    mu = 
