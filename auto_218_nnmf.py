# model file: ../example-models/misc/nnmf/nnmf.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'I' in data, 'variable not found in data: key=I'
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'X' in data, 'variable not found in data: key=X'
    assert 'sigma' in data, 'variable not found in data: key=sigma'
    # initialize data
    T = data["T"]
    I = data["I"]
    K = data["K"]
    X = data["X"]
    sigma = data["sigma"]
    check_constraints(T, low=0, dims=[1])
    check_constraints(I, low=0, dims=[1])
    check_constraints(K, low=0, dims=[1])
    check_constraints(X, dims=[T, I])
    check_constraints(sigma, low=0, dims=[I])

def transformed_data(data):
    # initialize data
    T = data["T"]
    I = data["I"]
    K = data["K"]
    X = data["X"]
    sigma = data["sigma"]
    g_bar = init_real("g_bar", low=0) # real/double
    g_sigma = init_real("g_sigma", low=0) # real/double
    alpha = init_vector("alpha", low=0, dims=(I)) # vector
    temp = init_vector("temp", dims=(T)) # vector
    for t in range(1, to_int(T) + 1):
        temp[t - 1] = _pyro_assign(temp[t - 1], _call_func("log", [_call_func("sum", [_index_select(X, t - 1) ])]))
    g_bar = _pyro_assign(g_bar, _call_func("mean", [temp]))
    g_sigma = _pyro_assign(g_sigma, _call_func("sd", [temp]))
    for i in range(1, to_int(I) + 1):
        alpha[i - 1] = _pyro_assign(alpha[i - 1], 10.0)
    data["g_bar"] = g_bar
    data["g_sigma"] = g_sigma
    data["alpha"] = alpha
    data["temp"] = temp

def init_params(data, params):
    # initialize data
    T = data["T"]
    I = data["I"]
    K = data["K"]
    X = data["X"]
    sigma = data["sigma"]
    # initialize transformed data
    g_bar = data["g_bar"]
    g_sigma = data["g_sigma"]
    alpha = data["alpha"]
    temp = data["temp"]
    # assign init values for parameters
    params["G"] = init_matrix("G", low=0, dims=(T, K)) # matrix
    params["F"] = init_simplex("F", dims=(K)) # real/double

def model(data, params):
    # initialize data
    T = data["T"]
    I = data["I"]
    K = data["K"]
    X = data["X"]
    sigma = data["sigma"]
    # initialize transformed data
    g_bar = data["g_bar"]
    g_sigma = data["g_sigma"]
    alpha = data["alpha"]
    temp = data["temp"]
    # INIT parameters
    G = params["G"]
    F = params["F"]
    # initialize transformed parameters
    # model block

    for t in range(1, to_int(T) + 1):
        G[t - 1] =  _pyro_sample(_index_select(G, t - 1) , "G[%d]" % (to_int(t-1)), "lognormal", [g_bar, g_sigma])
    for k in range(1, to_int(K) + 1):
        F[k - 1] =  _pyro_sample(_index_select(F, k - 1) , "F[%d]" % (to_int(k-1)), "dirichlet", [alpha])
    for t in range(1, to_int(T) + 1):

        for i in range(1, to_int(I) + 1):
            # {
            mu = init_real("mu") # real/double

            mu = _pyro_assign(mu, 0)
            for k in range(1, to_int(K) + 1):
                mu = _pyro_assign(mu, (mu + (_index_select(_index_select(G, t - 1) , k - 1)  * _index_select(_index_select(F, k - 1) , i - 1) )))
            X[t - 1][i - 1] =  _pyro_sample(_index_select(_index_select(X, t - 1) , i - 1) , "X[%d][%d]" % (to_int(t-1),to_int(i-1)), "normal", [mu, _index_select(sigma, i - 1) ], obs=_index_select(_index_select(X, t - 1) , i - 1) )
            # }

