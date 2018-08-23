# model file: ../example-models/bugs_examples/vol1/surgical/surgical.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'r' in data, 'variable not found in data: key=r'
    assert 'n' in data, 'variable not found in data: key=n'
    # initialize data
    N = data["N"]
    r = data["r"]
    n = data["n"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(r, dims=[N])
    check_constraints(n, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    r = data["r"]
    n = data["n"]
    # assign init values for parameters
    params["mu"] = init_real("mu") # real/double
    params["sigmasq"] = init_real("sigmasq", low=0) # real/double
    params["b"] = init_real("b", dims=(N)) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    r = data["r"]
    n = data["n"]
    # INIT parameters
    mu = params["mu"]
    sigmasq = params["sigmasq"]
    b = params["b"]
    # initialize transformed parameters
    sigma = init_real("sigma", low=0) # real/double
    p = init_real("p", low=0, high=1, dims=(N)) # real/double
    sigma = _pyro_assign(sigma, _call_func("sqrt", [sigmasq]))
    for i in range(1, to_int(N) + 1):
        p[i - 1] = _pyro_assign(p[i - 1], _call_func("inv_logit", [_index_select(b, i - 1) ]))
    # model block

    mu =  _pyro_sample(mu, "mu", "normal", [0.0, 1000.0])
    sigmasq =  _pyro_sample(sigmasq, "sigmasq", "inv_gamma", [0.001, 0.001])
    b =  _pyro_sample(b, "b", "normal", [mu, sigma])
    r =  _pyro_sample(r, "r", "binomial_logit", [n, b], obs=r)

