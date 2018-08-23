# model file: ../example-models/bugs_examples/vol1/mice/mice.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N_uncensored' in data, 'variable not found in data: key=N_uncensored'
    assert 'N_censored' in data, 'variable not found in data: key=N_censored'
    assert 'M' in data, 'variable not found in data: key=M'
    assert 'group_uncensored' in data, 'variable not found in data: key=group_uncensored'
    assert 'group_censored' in data, 'variable not found in data: key=group_censored'
    assert 'censor_time' in data, 'variable not found in data: key=censor_time'
    assert 't_uncensored' in data, 'variable not found in data: key=t_uncensored'
    # initialize data
    N_uncensored = data["N_uncensored"]
    N_censored = data["N_censored"]
    M = data["M"]
    group_uncensored = data["group_uncensored"]
    group_censored = data["group_censored"]
    censor_time = data["censor_time"]
    t_uncensored = data["t_uncensored"]
    check_constraints(N_uncensored, low=0, dims=[1])
    check_constraints(N_censored, low=0, dims=[1])
    check_constraints(M, low=0, dims=[1])
    check_constraints(group_uncensored, low=1, high=M, dims=[N_uncensored])
    check_constraints(group_censored, low=1, high=M, dims=[N_censored])
    check_constraints(censor_time, low=0, dims=[N_censored])
    check_constraints(t_uncensored, low=0, dims=[N_uncensored])

def init_params(data, params):
    # initialize data
    N_uncensored = data["N_uncensored"]
    N_censored = data["N_censored"]
    M = data["M"]
    group_uncensored = data["group_uncensored"]
    group_censored = data["group_censored"]
    censor_time = data["censor_time"]
    t_uncensored = data["t_uncensored"]
    # assign init values for parameters
    params["r"] = init_real("r", low=0) # real/double
    params["beta"] = init_real("beta", dims=(M)) # real/double
    params["t2_censored"] = init_real("t2_censored", low=1, dims=(N_censored)) # real/double

def model(data, params):
    # initialize data
    N_uncensored = data["N_uncensored"]
    N_censored = data["N_censored"]
    M = data["M"]
    group_uncensored = data["group_uncensored"]
    group_censored = data["group_censored"]
    censor_time = data["censor_time"]
    t_uncensored = data["t_uncensored"]
    # INIT parameters
    r = params["r"]
    beta = params["beta"]
    t2_censored = params["t2_censored"]
    # initialize transformed parameters
    # model block

    r =  _pyro_sample(r, "r", "exponential", [0.001])
    beta =  _pyro_sample(beta, "beta", "normal", [0, 100])
    for n in range(1, to_int(N_uncensored) + 1):

        t_uncensored[n - 1] =  _pyro_sample(_index_select(t_uncensored, n - 1) , "t_uncensored[%d]" % (to_int(n-1)), "weibull", [r, _call_func("exp", [(-(_index_select(beta, group_uncensored[n - 1] - 1) ) / r)])], obs=_index_select(t_uncensored, n - 1) )
    for n in range(1, to_int(N_censored) + 1):

        t2_censored[n - 1] =  _pyro_sample(_index_select(t2_censored, n - 1) , "t2_censored[%d]" % (to_int(n-1)), "weibull", [r, (_call_func("exp", [(-(_index_select(beta, group_censored[n - 1] - 1) ) / r)]) / _index_select(censor_time, n - 1) )])

