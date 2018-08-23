# model file: ../example-models/misc/hmm/hmm-sufficient.stan
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
    trans = init_int("trans", low=0, dims=(K, K)) # real/double
    emit = init_int("emit", low=0, dims=(K, V)) # real/double
    for k1 in range(1, to_int(K) + 1):
        for k2 in range(1, to_int(K) + 1):
            trans[k1 - 1][k2 - 1] = _pyro_assign(trans[k1 - 1][k2 - 1], 0)
    for t in range(2, to_int(T) + 1):
        trans[z[(t - 1) - 1] - 1][z[t - 1] - 1] = _pyro_assign(trans[z[(t - 1) - 1] - 1][z[t - 1] - 1], (1 + _index_select(_index_select(trans, z[(t - 1) - 1] - 1) , z[t - 1] - 1) ))
    for k in range(1, to_int(K) + 1):
        for v in range(1, to_int(V) + 1):
            emit[k - 1][v - 1] = _pyro_assign(emit[k - 1][v - 1], 0)
    for t in range(1, to_int(T) + 1):
        emit[z[t - 1] - 1][w[t - 1] - 1] = _pyro_assign(emit[z[t - 1] - 1][w[t - 1] - 1], (1 + _index_select(_index_select(emit, z[t - 1] - 1) , w[t - 1] - 1) ))
    data["trans"] = trans
    data["emit"] = emit

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
    trans = data["trans"]
    emit = data["emit"]
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
    trans = data["trans"]
    emit = data["emit"]
    # INIT parameters
    theta = params["theta"]
    phi = params["phi"]
    # initialize transformed parameters
    # model block

    for k in range(1, to_int(K) + 1):
        theta[k - 1] =  _pyro_sample(_index_select(theta, k - 1) , "theta[%d]" % (to_int(k-1)), "dirichlet", [alpha])
    for k in range(1, to_int(K) + 1):
        phi[k - 1] =  _pyro_sample(_index_select(phi, k - 1) , "phi[%d]" % (to_int(k-1)), "dirichlet", [beta])
    for k in range(1, to_int(K) + 1):
        trans[k - 1] =  _pyro_sample(_index_select(trans, k - 1) , "trans[%d]" % (to_int(k-1)), "multinomial", [_index_select(theta, k - 1) ], obs=_index_select(trans, k - 1) )
    for k in range(1, to_int(K) + 1):
        emit[k - 1] =  _pyro_sample(_index_select(emit, k - 1) , "emit[%d]" % (to_int(k-1)), "multinomial", [_index_select(phi, k - 1) ], obs=_index_select(emit, k - 1) )

