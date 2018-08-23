# model file: ../example-models/misc/hmm/hmm-analytic.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'V' in data, 'variable not found in data: key=V'
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'w' in data, 'variable not found in data: key=w'
    assert 'z' in data, 'variable not found in data: key=z'
    assert 'alpha' in data, 'variable not found in data: key=alpha'
    assert 'beta' in data, 'variable not found in data: key=beta'
    # initialize data
    K = data["K"]
    V = data["V"]
    T = data["T"]
    w = data["w"]
    z = data["z"]
    alpha = data["alpha"]
    beta = data["beta"]
    check_constraints(K, low=1, dims=[1])
    check_constraints(V, low=1, dims=[1])
    check_constraints(T, low=0, dims=[1])
    check_constraints(w, low=1, high=V, dims=[T])
    check_constraints(z, low=1, high=K, dims=[T])
    check_constraints(alpha, low=0, dims=[K])
    check_constraints(beta, low=0, dims=[V])

def transformed_data(data):
    # initialize data
    K = data["K"]
    V = data["V"]
    T = data["T"]
    w = data["w"]
    z = data["z"]
    alpha = data["alpha"]
    beta = data["beta"]
    alpha_post = init_vector("alpha_post", low=0, dims=(K, K)) # vector
    beta_post = init_vector("beta_post", low=0, dims=(K, V)) # vector
    for k in range(1, to_int(K) + 1):
        alpha_post[k - 1] = _pyro_assign(alpha_post[k - 1], alpha)
    for t in range(2, to_int(T) + 1):
        alpha_post[z[(t - 1) - 1] - 1][z[t - 1] - 1] = _pyro_assign(alpha_post[z[(t - 1) - 1] - 1][z[t - 1] - 1], (_index_select(_index_select(alpha_post, z[(t - 1) - 1] - 1) , z[t - 1] - 1)  + 1))
    for k in range(1, to_int(K) + 1):
        beta_post[k - 1] = _pyro_assign(beta_post[k - 1], beta)
    for t in range(1, to_int(T) + 1):
        beta_post[z[t - 1] - 1][w[t - 1] - 1] = _pyro_assign(beta_post[z[t - 1] - 1][w[t - 1] - 1], (_index_select(_index_select(beta_post, z[t - 1] - 1) , w[t - 1] - 1)  + 1))
    data["alpha_post"] = alpha_post
    data["beta_post"] = beta_post

def init_params(data, params):
    # initialize data
    K = data["K"]
    V = data["V"]
    T = data["T"]
    w = data["w"]
    z = data["z"]
    alpha = data["alpha"]
    beta = data["beta"]
    # initialize transformed data
    alpha_post = data["alpha_post"]
    beta_post = data["beta_post"]
    # assign init values for parameters
    params["theta"] = init_simplex("theta", dims=(K)) # real/double
    params["phi"] = init_simplex("phi", dims=(K)) # real/double

def model(data, params):
    # initialize data
    K = data["K"]
    V = data["V"]
    T = data["T"]
    w = data["w"]
    z = data["z"]
    alpha = data["alpha"]
    beta = data["beta"]
    # initialize transformed data
    alpha_post = data["alpha_post"]
    beta_post = data["beta_post"]
    # INIT parameters
    theta = params["theta"]
    phi = params["phi"]
    # initialize transformed parameters
    # model block

    for k in range(1, to_int(K) + 1):
        theta[k - 1] =  _pyro_sample(_index_select(theta, k - 1) , "theta[%d]" % (to_int(k-1)), "dirichlet", [_index_select(alpha_post, k - 1) ])
    for k in range(1, to_int(K) + 1):
        phi[k - 1] =  _pyro_sample(_index_select(phi, k - 1) , "phi[%d]" % (to_int(k-1)), "dirichlet", [_index_select(beta_post, k - 1) ])

