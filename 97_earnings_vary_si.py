# model file: ../example-models/ARM/Ch.13/earnings_vary_si.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'eth' in data, 'variable not found in data: key=eth'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(earn, dims=[N])
    check_constraints(eth, dims=[N])
    check_constraints(height, dims=[N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    log_earn = init_vector("log_earn", dims=(N)) # vector
    log_earn = _pyro_assign(log_earn, _call_func("log", [earn]))
    data["log_earn"] = log_earn

def init_params(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    # initialize transformed data
    log_earn = data["log_earn"]
    # assign init values for parameters
    params["a1"] = init_vector("a1", dims=(4)) # vector
    params["a2"] = init_vector("a2", dims=(4)) # vector
    params["sigma_a1"] = init_real("sigma_a1", low=0) # real/double
    params["sigma_a2"] = init_real("sigma_a2", low=0) # real/double
    params["sigma_y"] = init_real("sigma_y", low=0) # real/double
    params["mu_a1"] = init_real("mu_a1") # real/double
    params["mu_a2"] = init_real("mu_a2") # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    # initialize transformed data
    log_earn = data["log_earn"]
    # INIT parameters
    a1 = params["a1"]
    a2 = params["a2"]
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_y = params["sigma_y"]
    mu_a1 = params["mu_a1"]
    mu_a2 = params["mu_a2"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (_index_select(a1, eth[i - 1] - 1)  + (_index_select(a2, eth[i - 1] - 1)  * _index_select(height, i - 1) )))
    # model block

    mu_a1 =  _pyro_sample(mu_a1, "mu_a1", "normal", [0, 1])
    mu_a2 =  _pyro_sample(mu_a2, "mu_a2", "normal", [0, 1])
    a1 =  _pyro_sample(a1, "a1", "normal", [(10 * mu_a1), sigma_a1])
    a2 =  _pyro_sample(a2, "a2", "normal", [(0.01 * mu_a2), sigma_a2])
    sigma_a1 =  _pyro_sample(sigma_a1, "sigma_a1", "cauchy", [0, 5])
    sigma_a2 =  _pyro_sample(sigma_a2, "sigma_a2", "cauchy", [0, 5])
    sigma_y =  _pyro_sample(sigma_y, "sigma_y", "cauchy", [0, 5])
    log_earn =  _pyro_sample(log_earn, "log_earn", "normal", [y_hat, sigma_y], obs=log_earn)

