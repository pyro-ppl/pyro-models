# model file: ../example-models/basic_estimators/normal_mixture.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    y = data["y"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(y, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["theta"] = init_real("theta", low=0, high=1) # real/double
    params["mu"] = init_real("mu", dims=(2)) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # INIT parameters
    theta = params["theta"]
    mu = params["mu"]
    # initialize transformed parameters
    log_theta = init_real("log_theta") # real/double
    log_one_minus_theta = init_real("log_one_minus_theta") # real/double
    log_theta = _pyro_assign(log_theta, _call_func("log", [theta]))
    log_one_minus_theta = _pyro_assign(log_one_minus_theta, _call_func("log", [(1.0 - theta)]))
    # model block

    theta =  _pyro_sample(theta, "theta", "uniform", [0, 1])
    for k in range(1, 2 + 1):
        mu[k - 1] =  _pyro_sample(_index_select(mu, k - 1) , "mu[%d]" % (to_int(k-1)), "normal", [0, 10])
    for n in range(1, to_int(N) + 1):
        pyro.sample("_call_func( log_sum_exp , [(log_theta + _call_func( normal_log , [_index_select(y, n - 1) ,_index_select(mu, 1 - 1) ,1.0])),(log_one_minus_theta + _call_func( normal_log , [_index_select(y, n - 1) ,_index_select(mu, 2 - 1) ,1.0]))])[%d]" % (n), dist.Bernoulli(_call_func("log_sum_exp", [(log_theta + _call_func("normal_log", [_index_select(y, n - 1) ,_index_select(mu, 1 - 1) ,1.0])),(log_one_minus_theta + _call_func("normal_log", [_index_select(y, n - 1) ,_index_select(mu, 2 - 1) ,1.0]))])), obs=(1));

