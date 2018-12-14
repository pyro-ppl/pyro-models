# model file: ../example-models/bugs_examples/vol1/lsat/lsat.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'R' in data, 'variable not found in data: key=R'
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'culm' in data, 'variable not found in data: key=culm'
    assert 'response' in data, 'variable not found in data: key=response'
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(R, low=0, dims=[1])
    check_constraints(T, low=0, dims=[1])
    check_constraints(culm, low=0, dims=[R])
    check_constraints(response, low=0, dims=[R, T])

def transformed_data(data):
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]
    r = init_int("r", dims=(T, N)) # real/double
    ones = init_vector("ones", dims=(N)) # vector
    for j in range(1, to_int(culm[1 - 1]) + 1):

        for k in range(1, to_int(T) + 1):

            r[k - 1][j - 1] = _pyro_assign(r[k - 1][j - 1], _index_select(_index_select(response, 1 - 1) , k - 1) )
    for i in range(2, to_int(R) + 1):

        for j in range(to_int((culm[(i - 1) - 1] + 1)), to_int(culm[i - 1]) + 1):

            for k in range(1, to_int(T) + 1):

                r[k - 1][j - 1] = _pyro_assign(r[k - 1][j - 1], _index_select(_index_select(response, i - 1) , k - 1) )
    for i in range(1, to_int(N) + 1):
        ones[i - 1] = _pyro_assign(ones[i - 1], 1.0)
    data["r"] = r
    data["ones"] = ones

def init_params(data, params):
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]
    # initialize transformed data
    r = data["r"]
    ones = data["ones"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha", dims=(T)) # real/double
    params["theta"] = init_vector("theta", dims=(N)) # vector
    params["beta"] = init_real("beta", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]
    # initialize transformed data
    r = data["r"]
    ones = data["ones"]
    # INIT parameters
    alpha = params["alpha"]
    theta = params["theta"]
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0, 100.0])
    theta =  _pyro_sample(theta, "theta", "normal", [0, 1])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0, 100.0])
    for k in range(1, to_int(T) + 1):
        r[k - 1] =  _pyro_sample(_index_select(r, k - 1) , "r[%d]" % (to_int(k-1)), "bernoulli_logit", [_call_func("subtract", [_call_func("multiply", [beta,theta]),_call_func("multiply", [_index_select(alpha, k - 1) ,ones])])], obs=_index_select(r, k - 1) )

