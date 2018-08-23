# model file: ../example-models/ARM/Ch.16/radon.pooling.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'x' in data, 'variable not found in data: key=x'
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(y, dims=[N])
    check_constraints(x, low=0, high=1, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    # assign init values for parameters
    params["a"] = init_real("a") # real/double
    params["b"] = init_real("b") # real/double
    params["sigma_y"] = init_real("sigma_y", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    # INIT parameters
    a = params["a"]
    b = params["b"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    # model block

    y =  _pyro_sample(y, "y", "normal", [_call_func("add", [a,_call_func("multiply", [b,x])]), sigma_y], obs=y)

