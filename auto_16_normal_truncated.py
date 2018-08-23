# model file: ../example-models/basic_estimators/normal_truncated.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'U' in data, 'variable not found in data: key=U'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    U = data["U"]
    N = data["N"]
    y = data["y"]
    check_constraints(U, dims=[1])
    check_constraints(N, low=1, dims=[1])
    check_constraints(y, high=U, dims=[N])

def init_params(data, params):
    # initialize data
    U = data["U"]
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["mu"] = init_real("mu") # real/double
    params["sigma"] = init_real("sigma", low=0, high=2) # real/double

def model(data, params):
    # initialize data
    U = data["U"]
    N = data["N"]
    y = data["y"]
    # INIT parameters
    mu = params["mu"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    for n in range(1, to_int(N) + 1):
        y[n - 1] =  _pyro_sample(_index_select(y, n - 1) , "y[%d]" % (to_int(n-1)), "normal", [mu, sigma], obs=_index_select(y, n - 1) )

