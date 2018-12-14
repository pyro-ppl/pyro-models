# model file: ../example-models/ARM/Ch.17/17.3_flight_simulator.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_treatment' in data, 'variable not found in data: key=n_treatment'
    assert 'n_airport' in data, 'variable not found in data: key=n_airport'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'airport' in data, 'variable not found in data: key=airport'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_treatment = data["n_treatment"]
    n_airport = data["n_airport"]
    treatment = data["treatment"]
    airport = data["airport"]
    y = data["y"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(n_treatment, low=0, dims=[1])
    check_constraints(n_airport, low=0, dims=[1])
    check_constraints(treatment, low=0, high=n_treatment, dims=[N])
    check_constraints(airport, low=0, high=n_airport, dims=[N])
    check_constraints(y, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_treatment = data["n_treatment"]
    n_airport = data["n_airport"]
    treatment = data["treatment"]
    airport = data["airport"]
    y = data["y"]
    # assign init values for parameters
    params["sigma"] = init_real("sigma", low=0) # real/double
    params["sigma_gamma"] = init_real("sigma_gamma", low=0) # real/double
    params["sigma_delta"] = init_real("sigma_delta", low=0) # real/double
    params["gamma"] = init_vector("gamma", dims=(n_treatment)) # vector
    params["delta"] = init_vector("delta", dims=(n_airport)) # vector
    params["mu"] = init_real("mu") # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n_treatment = data["n_treatment"]
    n_airport = data["n_airport"]
    treatment = data["treatment"]
    airport = data["airport"]
    y = data["y"]
    # INIT parameters
    sigma = params["sigma"]
    sigma_gamma = params["sigma_gamma"]
    sigma_delta = params["sigma_delta"]
    gamma = params["gamma"]
    delta = params["delta"]
    mu = params["mu"]
    # initialize transformed parameters
    # model block
    # {
    y_hat = init_vector("y_hat", dims=(N)) # vector

    sigma =  _pyro_sample(sigma, "sigma", "uniform", [0, 100])
    sigma_gamma =  _pyro_sample(sigma_gamma, "sigma_gamma", "uniform", [0, 100])
    sigma_delta =  _pyro_sample(sigma_delta, "sigma_delta", "uniform", [0, 100])
    mu =  _pyro_sample(mu, "mu", "normal", [0, 100])
    gamma =  _pyro_sample(gamma, "gamma", "normal", [0, sigma_gamma])
    delta =  _pyro_sample(delta, "delta", "normal", [0, sigma_delta])
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], ((mu + _index_select(gamma, treatment[i - 1] - 1) ) + _index_select(delta, airport[i - 1] - 1) ))
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma], obs=y)
    # }

