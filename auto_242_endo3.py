# model file: ../example-models/bugs_examples/vol2/endo/endo3.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n10' in data, 'variable not found in data: key=n10'
    assert 'n01' in data, 'variable not found in data: key=n01'
    assert 'n11' in data, 'variable not found in data: key=n11'
    assert 'I' in data, 'variable not found in data: key=I'
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    check_constraints(n10, dims=[1])
    check_constraints(n01, dims=[1])
    check_constraints(n11, dims=[1])
    check_constraints(I, dims=[1])

def transformed_data(data):
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    J = init_int("J") # real/double
    Y = init_int("Y", low=0, dims=(2, I)) # real/double
    est = init_vector("est", low=0, dims=(2, I)) # vector
    J = _pyro_assign(J, 2)
    for i in range(1, to_int(I) + 1):

        Y[1 - 1][i - 1] = _pyro_assign(Y[1 - 1][i - 1], 1)
        Y[2 - 1][i - 1] = _pyro_assign(Y[2 - 1][i - 1], 0)
    for i in range(1, to_int(n10) + 1):

        est[1 - 1][i - 1] = _pyro_assign(est[1 - 1][i - 1], 1)
        est[2 - 1][i - 1] = _pyro_assign(est[2 - 1][i - 1], 0)
    for i in range(to_int((n10 + 1)), to_int((n10 + n01)) + 1):

        est[1 - 1][i - 1] = _pyro_assign(est[1 - 1][i - 1], 0)
        est[2 - 1][i - 1] = _pyro_assign(est[2 - 1][i - 1], 1)
    for i in range(to_int(((n10 + n01) + 1)), to_int(((n10 + n01) + n11)) + 1):

        est[1 - 1][i - 1] = _pyro_assign(est[1 - 1][i - 1], 1)
        est[2 - 1][i - 1] = _pyro_assign(est[2 - 1][i - 1], 1)
    for i in range(to_int((((n10 + n01) + n11) + 1)), to_int(I) + 1):

        est[1 - 1][i - 1] = _pyro_assign(est[1 - 1][i - 1], 0)
        est[2 - 1][i - 1] = _pyro_assign(est[2 - 1][i - 1], 0)
    data["J"] = J
    data["Y"] = Y
    data["est"] = est

def init_params(data, params):
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    # initialize transformed data
    J = data["J"]
    Y = data["Y"]
    est = data["est"]
    # assign init values for parameters
    params["beta"] = init_real("beta") # real/double
    params["beta0"] = init_vector("beta0", dims=(I)) # vector

def model(data, params):
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    # initialize transformed data
    J = data["J"]
    Y = data["Y"]
    est = data["est"]
    # INIT parameters
    beta = params["beta"]
    beta0 = params["beta0"]
    # initialize transformed parameters
    # model block

    beta =  _pyro_sample(beta, "beta", "normal", [0, 1000])
    beta0 =  _pyro_sample(beta0, "beta0", "normal", [0, 1000])
    for j in range(1, to_int(J) + 1):
        Y[j - 1] =  _pyro_sample(_index_select(Y, j - 1) , "Y[%d]" % (to_int(j-1)), "poisson_log", [_call_func("add", [beta0,_call_func("multiply", [beta,_index_select(est, j - 1) ])])], obs=_index_select(Y, j - 1) )

