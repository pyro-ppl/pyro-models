# model file: ../example-models/misc/moving-avg/arma11.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    T = data["T"]
    y = data["y"]
    check_constraints(T, low=1, dims=[1])
    check_constraints(y, dims=[T])

def init_params(data, params):
    # initialize data
    T = data["T"]
    y = data["y"]
    # assign init values for parameters
    params["mu"] = init_real("mu") # real/double
    params["phi"] = init_real("phi") # real/double
    params["theta"] = init_real("theta") # real/double
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    T = data["T"]
    y = data["y"]
    # INIT parameters
    mu = params["mu"]
    phi = params["phi"]
    theta = params["theta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block
    # {
    nu = init_vector("nu", dims=(T)) # vector
    err = init_vector("err", dims=(T)) # vector

    nu[1 - 1] = _pyro_assign(nu[1 - 1], (mu + (phi * mu)))
    err[1 - 1] = _pyro_assign(err[1 - 1], (_index_select(y, 1 - 1)  - _index_select(nu, 1 - 1) ))
    for t in range(2, to_int(T) + 1):

        nu[t - 1] = _pyro_assign(nu[t - 1], ((mu + (phi * _index_select(y, (t - 1) - 1) )) + (theta * _index_select(err, (t - 1) - 1) )))
        err[t - 1] = _pyro_assign(err[t - 1], (_index_select(y, t - 1)  - _index_select(nu, t - 1) ))
    mu =  _pyro_sample(mu, "mu", "normal", [0, 10])
    phi =  _pyro_sample(phi, "phi", "normal", [0, 2])
    theta =  _pyro_sample(theta, "theta", "normal", [0, 2])
    sigma =  _pyro_sample(sigma, "sigma", "cauchy", [0, 5])
    err =  _pyro_sample(err, "err", "normal", [0, sigma])
    # }

