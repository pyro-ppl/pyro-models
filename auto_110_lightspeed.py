# model file: ../example-models/ARM/Ch.8/lightspeed.stan
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
    params["beta"] = init_vector("beta", dims=(1)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    y =  _pyro_sample(y, "y", "normal", [_index_select(beta, 1 - 1) , sigma], obs=y)

