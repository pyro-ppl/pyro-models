# model file: ../example-models/BPA/Ch.07/cjs_mnl_ran.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n_occasions' in data, 'variable not found in data: key=n_occasions'
    assert 'marr' in data, 'variable not found in data: key=marr'
    # initialize data
    n_occasions = data["n_occasions"]
    marr = data["marr"]
    check_constraints(n_occasions, low=0, dims=[1])
    check_constraints(marr, low=0, dims=[(n_occasions - 1), n_occasions])

def transformed_data(data):
    # initialize data
    n_occasions = data["n_occasions"]
    marr = data["marr"]
    n_occ_minus_1 = init_int("n_occ_minus_1") # real/double
    r = init_int("r", dims=(n_occasions)) # real/double
    for t in range(1, to_int(n_occ_minus_1) + 1):
        r[t - 1] = _pyro_assign(r[t - 1], _call_func("sum", [_index_select(marr, t - 1) ]))
    data["n_occ_minus_1"] = n_occ_minus_1
    data["r"] = r

def init_params(data, params):
    # initialize data
    n_occasions = data["n_occasions"]
    marr = data["marr"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    r = data["r"]
    # assign init values for parameters
    params["mean_phi"] = init_real("mean_phi", low=0, high=1) # real/double
    params["mean_p"] = init_real("mean_p", low=0, high=1) # real/double
    params["sigma"] = init_real("sigma", low=0) # real/double
    params["epsilon"] = init_vector("epsilon", dims=(n_occ_minus_1)) # vector

def model(data, params):
    # initialize data
    n_occasions = data["n_occasions"]
    marr = data["marr"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    r = data["r"]
    # INIT parameters
    mean_phi = params["mean_phi"]
    mean_p = params["mean_p"]
    sigma = params["sigma"]
    epsilon = params["epsilon"]
    # initialize transformed parameters
    mu = init_real("mu") # real/double
    phi = init_vector("phi", low=0, high=1, dims=(n_occ_minus_1)) # vector
    p = init_vector("p", low=0, high=1, dims=(n_occ_minus_1)) # vector
    q = init_vector("q", low=0, high=1, dims=(n_occ_minus_1)) # vector
    pr = init_simplex("pr", dims=(n_occ_minus_1)) # real/double
    mu = _pyro_assign(mu, _call_func("logit", [mean_phi]))
    phi = _pyro_assign(phi, _call_func("inv_logit", [_call_func("add", [mu,epsilon])]))
    p = _pyro_assign(p, _call_func("rep_vector", [mean_p,n_occ_minus_1]))
    q = _pyro_assign(q, _call_func("subtract", [1.0,p]))
    for t in range(1, to_int(n_occ_minus_1) + 1):

        pr[t - 1][t - 1] = _pyro_assign(pr[t - 1][t - 1], (_index_select(phi, t - 1)  * _index_select(p, t - 1) ))
        for j in range(to_int((t + 1)), to_int(n_occ_minus_1) + 1):
            pr[t - 1][j - 1] = _pyro_assign(pr[t - 1][j - 1], ((_call_func("prod", [
