# model file: ../example-models/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial/Survey.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'nmax' in data, 'variable not found in data: key=nmax'
    assert 'm' in data, 'variable not found in data: key=m'
    assert 'k' in data, 'variable not found in data: key=k'
    # initialize data
    nmax = data["nmax"]
    m = data["m"]
    k = data["k"]
    check_constraints(nmax, low=0, dims=[1])
    check_constraints(m, low=0, dims=[1])
    check_constraints(k, low=0, high=nmax, dims=[m])

def transformed_data(data):
    # initialize data
    nmax = data["nmax"]
    m = data["m"]
    k = data["k"]
    nmin = init_int("nmin", low=0) # real/double
    nmin = _pyro_assign(nmin, _call_func("max", [k]))
    data["nmin"] = nmin

def init_params(data, params):
    # initialize data
    nmax = data["nmax"]
    m = data["m"]
    k = data["k"]
    # initialize transformed data
    nmin = data["nmin"]
    # assign init values for parameters
    params["theta"] = init_real("theta", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    nmax = data["nmax"]
    m = data["m"]
    k = data["k"]
    # initialize transformed data
    nmin = data["nmin"]
    # INIT parameters
    theta = params["theta"]
    # initialize transformed parameters
    lp_parts = init_vector("lp_parts", dims=(nmax)) # vector
    for n in range(1, to_int(nmax) + 1):
        if (as_bool(_call_func("logical_lt", [n,nmin]))):
            lp_parts[n - 1] = _pyro_assign(lp_parts[n - 1], (_call_func("log", [(1.0 / nmax)]) + _call_func("negative_infinity", [])))
        else: 
            lp_parts[n - 1] = _pyro_assign(lp_parts[n - 1], (_call_func("log", [(1.0 / nmax)]) + _call_func("binomial_log", [k,n,theta])))
        
    # model block

    pyro.sample("_call_func( log_sum_exp , [lp_parts])", dist.Bernoulli(_call_func("log_sum_exp", [lp_parts])), obs=(1));

