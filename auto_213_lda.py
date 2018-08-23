# model file: ../example-models/misc/cluster/lda/lda.stan
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
    params["theta"] = init_simplex("theta", dims=(M)) # real/double
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

    for m in range(1, to_int(M) + 1):
        theta[m - 1] =  _pyro_sample(_index_select(theta, m - 1) , "theta[%d]" % (to_int(m-1)), "dirichlet", [alpha])
    for k in range(1, to_int(K) + 1):
        phi[k - 1] =  _pyro_sample(_index_select(phi, k - 1) , "phi[%d]" % (to_int(k-1)), "dirichlet", [beta])
    for n in range(1, to_int(N) + 1):
        # {
        gamma = init_real("gamma", dims=(K)) # real/double

        for k in range(1, to_int(K) + 1):
            gamma[k - 1] = _pyro_assign(gamma[k - 1], (_call_func("log", [_index_select(_index_select(theta, doc[n - 1] - 1) , k - 1) ]) + _call_func("log", [_index_select(_index_select(phi, k - 1) , w[n - 1] - 1) ])))
        pyro.sample("_call_func( log_sum_exp , [gamma])[%d]" % (n), dist.Bernoulli(_call_func("log_sum_exp", [gamma])), obs=(1));
        # }

