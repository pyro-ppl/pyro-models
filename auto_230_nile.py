# model file: ../example-models/misc/dlm/nile.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'm0' in data, 'variable not found in data: key=m0'
    assert 'C0' in data, 'variable not found in data: key=C0'
    # initialize data
    n = data["n"]
    y = data["y"]
    m0 = data["m0"]
    C0 = data["C0"]
    check_constraints(n, dims=[1])
    check_constraints(y, dims=[1, n])
    check_constraints(m0, dims=[1])
    check_constraints(C0, dims=[1, 1])

def transformed_data(data):
    # initialize data
    n = data["n"]
    y = data["y"]
    m0 = data["m0"]
    C0 = data["C0"]
    F = init_matrix("F", dims=(1, 1)) # matrix
    G = init_matrix("G", dims=(1, 1)) # matrix
    F = _pyro_assign(F, _call_func("rep_matrix", [1,1,1]))
    G = _pyro_assign(G, _call_func("rep_matrix", [1,1,1]))
    data["F"] = F
    data["G"] = G

def init_params(data, params):
    # initialize data
    n = data["n"]
    y = data["y"]
    m0 = data["m0"]
    C0 = data["C0"]
    # initialize transformed data
    F = data["F"]
    G = data["G"]
    # assign init values for parameters
    params["sigma_y"] = init_real("sigma_y", low=0) # real/double
    params["sigma_theta"] = init_real("sigma_theta", low=0) # real/double

def model(data, params):
    # initialize data
    n = data["n"]
    y = data["y"]
    m0 = data["m0"]
    C0 = data["C0"]
    # initialize transformed data
    F = data["F"]
    G = data["G"]
    # INIT parameters
    sigma_y = params["sigma_y"]
    sigma_theta = params["sigma_theta"]
    # initialize transformed parameters
    # model block
    # {
    V = init_matrix("V", dims=(1, 1)) # matrix
    W = init_matrix("W", dims=(1, 1)) # matrix

    V[1 - 1][1 - 1] = _pyro_assign(V[1 - 1][1 - 1], _call_func("pow", [sigma_y,2]))
    W[1 - 1][1 - 1] = _pyro_assign(W[1 - 1][1 - 1], _call_func("pow", [sigma_theta,2]))
    y =  _pyro_sample(y, "y", "gaussian_dlm_obs", [F, G, V, W, m0, C0], obs=y)
    # }

