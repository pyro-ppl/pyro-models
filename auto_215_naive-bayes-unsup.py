# model file: ../example-models/misc/cluster/naive-bayes/naive-bayes-unsup.stan
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
    assert 'alpha' in data, 'variable not found in data: key=alpha'
    assert 'beta' in data, 'variable not found in data: key=beta'
    # initialize data
    K = data["K"]
    V = data["V"]
    M = data["M"]
    N = data["N"]
    w = data["w"]
    doc = data["doc"]
    alpha = data["alpha"]
    beta = data["beta"]
    check_constraints(K, low=2, dims=[1])
    check_constraints(V, low=2, dims=[1])
    check_constraints(M, low=1, dims=[1])
    check_constraints(N, low=1, dims=[1])
    check_constraints(w, low=1, high=V, dims=[N])
    check_constraints(doc, low=1, high=M, dims=[N])
    check_constraints(alpha, low=0, dims=[K])
    check_constraints(beta, low=0, dims=[V])

def init_params(data, params):
    # initialize data
    K = data["K"]
    V = data["V"]
    M = data["M"]
    N = data["N"]
    w = data["w"]
    doc = data["doc"]
    alpha = data["alpha"]
    beta = data["beta"]
    # assign init values for parameters
    params["theta"] = init_simplex("theta") # real/double
    params["phi"] = init_simplex("phi", dims=(K)) # real/double

def model(data, params):
    # initialize data
    K = data["K"]
    V = data["V"]
    M = data["M"]
    N = data["N"]
    w = data["w"]
    doc = data["doc"]
    alpha = data["alpha"]
    beta = data["beta"]
    # INIT parameters
    theta = params["theta"]
    phi = params["phi"]
    # initialize transformed parameters
    # model block
    # {
    gamma = init_real("gamma", dims=(M, K)) # real/double

    theta =  _pyro_sample(theta, "theta", "dirichlet", [alpha])
    for k in range(1, to_int(K) + 1):
        phi[k - 1] =  _pyro_sample(_index_select(phi, k - 1) , "phi[%d]" % (to_int(k-1)), "dirichlet", [beta])
    for m in range(1, to_int(M) + 1):
        for k in range(1, to_int(K) + 1):
            gamma[m - 1][k - 1] = _pyro_assign(gamma[m - 1][k - 1], _call_func("categorical_log", [k,theta]))
    for n in range(1, to_int(N) + 1):
        for k in range(1, to_int(K) + 1):
            gamma[doc[n - 1] - 1][k - 1] = _pyro_assign(gamma[doc[n - 1] - 1][k - 1], (_index_select(_index_select(gamma, doc[n - 1] - 1) , k - 1)  + _call_func("categorical_log", [_index_select(w, n - 1) ,_index_select(phi, k - 1) ])))
    for m in range(1, to_int(M) + 1):
        pyro.sample("_call_func( log_sum_exp , [_index_select(gamma, m - 1) ])[%d]" % (m), dist.Bernoulli(_call_func("log_sum_exp", [_index_select(gamma, m - 1) ])), obs=(1));
    # }

