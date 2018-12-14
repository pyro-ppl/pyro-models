# model file: ../example-models/ARM/Ch.4/log10earn_height.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(earn, dims=[N])
    check_constraints(height, dims=[N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    log10_earn = init_vector("log10_earn", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):

        log10_earn[i - 1] = _pyro_assign(log10_earn[i - 1], _call_func("log10", [_index_select(earn, i - 1) ]))
    data["log10_earn"] = log10_earn

def init_params(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    # initialize transformed data
    log10_earn = data["log10_earn"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    # initialize transformed data
    log10_earn = data["log10_earn"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log10_earn =  _pyro_sample(log10_earn, "log10_earn", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,height])]), sigma], obs=log10_earn)

