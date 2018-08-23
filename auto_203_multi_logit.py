# model file: ../example-models/misc/multi-logit/multi_logit.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'D' in data, 'variable not found in data: key=D'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    K = data["K"]
    D = data["D"]
    N = data["N"]
    x = data["x"]
    y = data["y"]
    check_constraints(K, low=2, dims=[1])
    check_constraints(D, low=2, dims=[1])
    check_constraints(N, low=0, dims=[1])
    check_constraints(x, dims=[N, D])
    check_constraints(y, low=1, high=K, dims=[N])

def init_params(data, params):
    # initialize data
    K = data["K"]
    D = data["D"]
    N = data["N"]
    x = data["x"]
    y = data["y"]
    # assign init values for parameters
    params["beta"] = init_matrix("beta", dims=(K, D)) # matrix

def model(data, params):
    # initialize data
    K = data["K"]
    D = data["D"]
    N = data["N"]
    x = data["x"]
    y = data["y"]
    # INIT parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block
    # {
    gamma = init_matrix("gamma", dims=(N, K)) # matrix

    gamma = _pyro_assign(gamma, _call_func("multiply", [x,_call_func("transpose", [beta])]))
    _call_func("to_vector", [beta]) =  _pyro_sample(_call_func("to_vector", [beta]), "_call_func( to_vector , [beta])", "cauchy", [0, 2.5])
    for n in range(1, to_int(N) + 1):
        y[n - 1] =  _pyro_sample(_index_select(y, n - 1) , "y[%d]" % (to_int(n-1)), "categorical_logit", [_call_func("transpose", [_index_select(gamma, n - 1) ])], obs=_index_select(y, n - 1) )
    # }

