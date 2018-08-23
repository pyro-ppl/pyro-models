# model file: ../example-models/misc/hmm/hmm.stan
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

def init_params(data, params):
    # initialize data
    K = data["K"]
    V = data["V"]
    T = data["T"]
    w = data["w"]
    z = data["z"]
    alpha = data["alpha"]
    beta = data["beta"]
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
    # INIT parameters
    theta = params["theta"]
    phi = params["phi"]
    # initialize transformed parameters
    # model block

    for k in range(1, to_int(K) + 1):
        theta[k - 1] =  _pyro_sample(_index_select(theta, k - 1) , "theta[%d]" % (to_int(k-1)), "dirichlet", [alpha])
    for k in range(1, to_int(K) + 1):
        phi[k - 1] =  _pyro_sample(_index_select(phi, k - 1) , "phi[%d]" % (to_int(k-1)), "dirichlet", [beta])
    for t in range(1, to_int(T) + 1):
        w[t - 1] =  _pyro_sample(_index_select(w, t - 1) , "w[%d]" % (to_int(t-1)), "categorical", [_index_select(phi, z[t - 1] - 1) ], obs=_index_select(w, t - 1) )
    for t in range(2, to_int(T) + 1):
        z[t - 1] =  _pyro_sample(_index_select(z, t - 1) , "z[%d]" % (to_int(t-1)), "categorical", [_index_select(theta, z[(t - 1) - 1] - 1) ], obs=_index_select(z, t - 1) )

