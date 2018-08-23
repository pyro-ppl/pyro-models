# model file: ../example-models/misc/cluster/lda/corr-lda.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'V' in data, 'variable not found in data: key=V'
    assert 'M' in data, 'variable not found in data: key=M'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'w' in data, 'variable not found in data: key=w'
    assert 'doc' in data, 'variable not found in data: key=doc'
    assert 'beta' in data, 'variable not found in data: key=beta'
    # initialize data
    K = data["K"]
    V = data["V"]
    M = data["M"]
    N = data["N"]
    w = data["w"]
    doc = data["doc"]
    beta = data["beta"]
    check_constraints(K, low=2, dims=[1])
    check_constraints(V, low=2, dims=[1])
    check_constraints(M, low=1, dims=[1])
    check_constraints(N, low=1, dims=[1])
    check_constraints(w, low=1, high=V, dims=[N])
    check_constraints(doc, low=1, high=M, dims=[N])
    check_constraints(beta, low=0, dims=[V])

def init_params(data, params):
    # initialize data
    K = data["K"]
    V = data["V"]
    M = data["M"]
    N = data["N"]
    w = data["w"]
    doc = data["doc"]
    beta = data["beta"]
    # assign init values for parameters
    params["mu"] = init_vector("mu", dims=(K)) # vector
    params["Omega"] = 
