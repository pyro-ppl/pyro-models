# model file: ../example-models/bugs_examples/vol2/stagnant/stagnant2.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'Y' in data, 'variable not found in data: key=Y'
    # initialize data
    N = data["N"]
    x = data["x"]
    Y = data["Y"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(x, dims=[N])
    check_constraints(Y, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    Y = data["Y"]
    # assign init values for parameters
    params["sigma"] = init_real("sigma", low=0) # real/double
    params["alpha"] = init_real("alpha", low=0) # real/double
    params["beta"] = init_real("beta", dims=(2)) # real/double
    params["x_change"] = init_real("x_change", low=_call_func("min", [x]), high=_call_func("max", [x])) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    Y = data["Y"]
    # INIT parameters
    sigma = params["sigma"]
    alpha = params["alpha"]
    beta = params["beta"]
    x_change = params["x_change"]
    # initialize transformed parameters
    # model block
    # {
    mu = init_real("mu", dims=(N)) # real/double

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0, 5])
    beta =  _pyro_sample(beta, "beta", "normal", [0, 5])
    sigma =  _pyro_sample(sigma, "sigma", "cauchy", [0, 5])
    for n in range(1, to_int(N) + 1):
        mu[n - 1] = _pyro_assign(mu[n - 1], (alpha + (_call_func("if_else", [_call_func("logical_lt", [_index_select(x, n - 1) ,x_change]),_index_select(beta, 1 - 1) ,_index_select(beta, 2 - 1) ]) * (_index_select(x, n - 1)  - x_change))))
    Y =  _pyro_sample(Y, "Y", "normal", [mu, sigma], obs=Y)
    # }

