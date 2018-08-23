# model file: ../example-models/BPA/Ch.10/js_super_indran.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'M' in data, 'variable not found in data: key=M'
    assert 'n_occasions' in data, 'variable not found in data: key=n_occasions'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    M = data["M"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    check_constraints(M, low=0, dims=[1])
    check_constraints(n_occasions, low=0, dims=[1])
    check_constraints(y, low=0, high=1, dims=[M, n_occasions])

def transformed_data(data):
    # initialize data
    M = data["M"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    first = init_int("first", low=0, high=n_occasions, dims=(M)) # real/double
    last = init_int("last", low=0, high=n_occasions, dims=(M)) # real/double
    for i in range(1, to_int(M) + 1):
        first[i - 1] = _pyro_assign(first[i - 1], _call_func("first_capture", [_index_select(y, i - 1) , pstream__]))
    for i in range(1, to_int(M) + 1):
        last[i - 1] = _pyro_assign(last[i - 1], _call_func("last_capture", [_index_select(y, i - 1) , pstream__]))
    data["first"] = first
    data["last"] = last

def init_params(data, params):
    # initialize data
    M = data["M"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    first = data["first"]
    last = data["last"]
    # assign init values for parameters
    params["mean_phi"] = init_real("mean_phi", low=0, high=1) # real/double
    params["mean_p"] = init_real("mean_p", low=0, high=1) # real/double
    params["psi"] = init_real("psi", low=0, high=1) # real/double
    params["beta"] = init_vector("beta", low=0, dims=(n_occasions)) # vector
    params["epsilon"] = init_vector("epsilon", dims=(M)) # vector
    params["sigma"] = init_real("sigma", low=0, high=5) # real/double

def model(data, params):
    # initialize data
    M = data["M"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    first = data["first"]
    last = data["last"]
    # INIT parameters
    mean_phi = params["mean_phi"]
    mean_p = params["mean_p"]
    psi = params["psi"]
    beta = params["beta"]
    epsilon = params["epsilon"]
    sigma = params["sigma"]
    # initialize transformed parameters
    phi = init_matrix("phi", low=0, high=1, dims=(M, (n_occasions - 1))) # matrix
    p = init_matrix("p", low=0, high=1, dims=(M, n_occasions)) # matrix
    b = init_simplex("b") # real/double
    nu = init_vector("nu", low=0, high=1, dims=(n_occasions)) # vector
    chi = init_matrix("chi", low=0, high=1, dims=(M, n_occasions)) # matrix
    phi = _pyro_assign(phi, _call_func("rep_matrix", [mean_phi,M,(n_occasions - 1)]))
    for t in range(1, to_int(n_occasions) + 1):
        p[:, t] = _call_func("inv_logit", [_call_func("add", [_call_func("logit", [mean_p]),epsilon])])
    b = _pyro_assign(b, _call_func("divide", [beta,_call_func("sum", [beta])]))
    # {
    cum_b = init_real("cum_b") # real/double

    nu[1 - 1] = _pyro_assign(nu[1 - 1], _index_select(b, 1 - 1) )
    for t in range(2, to_int((n_occasions - 1)) + 1):

        nu[t - 1] = _pyro_assign(nu[t - 1], (_index_select(b, t - 1)  / (1.0 - cum_b)))
        cum_b = _pyro_assign(cum_b, (cum_b + _index_select(b, t - 1) ))
    nu[n_occasions - 1] = _pyro_assign(nu[n_occasions - 1], 1.0)
    # }
    chi = _pyro_assign(chi, _call_func("prob_uncaptured", [p,phi, pstream__]))
    # model block

    epsilon =  _pyro_sample(epsilon, "epsilon", "normal", [0, sigma])
    beta =  _pyro_sample(beta, "beta", "gamma", [1, 1])
    _call_func("js_super_lp", [y,first,last,p,phi,psi,nu,chi, lp__, lp_accum__, pstream__]);

