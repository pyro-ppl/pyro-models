# model file: ../example-models/bugs_examples/vol2/alli/alli2.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'I' in data, 'variable not found in data: key=I'
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'X' in data, 'variable not found in data: key=X'
    # initialize data
    I = data["I"]
    J = data["J"]
    K = data["K"]
    X = data["X"]
    check_constraints(I, dims=[1])
    check_constraints(J, dims=[1])
    check_constraints(K, dims=[1])
    check_constraints(X, dims=[I, J, K])

def init_params(data, params):
    # initialize data
    I = data["I"]
    J = data["J"]
    K = data["K"]
    X = data["X"]
    # assign init values for parameters
    params["alpha0"] = init_vector("alpha0", dims=((K - 1))) # vector
    params["beta0"] = init_matrix("beta0", dims=((I - 1), (K - 1))) # matrix
    params["gamma0"] = init_matrix("gamma0", dims=((J - 1), (K - 1))) # matrix
    params["lambda_"] = init_matrix("lambda_", dims=(I, J)) # matrix

def model(data, params):
    # initialize data
    I = data["I"]
    J = data["J"]
    K = data["K"]
    X = data["X"]
    # INIT parameters
    alpha0 = params["alpha0"]
    beta0 = params["beta0"]
    gamma0 = params["gamma0"]
    lambda_ = params["lambda_"]
    # initialize transformed parameters
    alpha = init_vector("alpha", dims=(K)) # vector
    beta = init_vector("beta", dims=(I, K)) # vector
    gamma = init_vector("gamma", dims=(J, K)) # vector
    alpha[1 - 1] = _pyro_assign(alpha[1 - 1], 0)
    for k in range(1, to_int((K - 1)) + 1):
        alpha[(k + 1) - 1] = _pyro_assign(alpha[(k + 1) - 1], _index_select(alpha0, k - 1) )
    for i in range(1, to_int(I) + 1):
        beta[i - 1][1 - 1] = _pyro_assign(beta[i - 1][1 - 1], 0)
    for k in range(1, to_int(K) + 1):
        beta[1 - 1][k - 1] = _pyro_assign(beta[1 - 1][k - 1], 0)
    for i in range(1, to_int((I - 1)) + 1):
        for k in range(1, to_int((K - 1)) + 1):
            beta[(i + 1) - 1][(k + 1) - 1] = _pyro_assign(beta[(i + 1) - 1][(k + 1) - 1], _index_select(_index_select(beta0, i - 1) , k - 1) )
    for j in range(1, to_int(J) + 1):
        gamma[j - 1][1 - 1] = _pyro_assign(gamma[j - 1][1 - 1], 0)
    for k in range(1, to_int(K) + 1):
        gamma[1 - 1][k - 1] = _pyro_assign(gamma[1 - 1][k - 1], 0)
    for j in range(1, to_int((J - 1)) + 1):
        for k in range(1, to_int((K - 1)) + 1):
            gamma[(j + 1) - 1][(k + 1) - 1] = _pyro_assign(gamma[(j + 1) - 1][(k + 1) - 1], _index_select(_index_select(gamma0, j - 1) , k - 1) )
    # model block

    for k in range(2, to_int(K) + 1):

        alpha[k - 1] =  _pyro_sample(_index_select(alpha, k - 1) , "alpha[%d]" % (to_int(k-1)), "normal", [0, 320])
        for i in range(2, to_int(I) + 1):
            beta[i - 1][k - 1] =  _pyro_sample(_index_select(_index_select(beta, i - 1) , k - 1) , "beta[%d][%d]" % (to_int(i-1),to_int(k-1)), "normal", [0, 320])
        for j in range(2, to_int(J) + 1):
            gamma[j - 1][k - 1] =  _pyro_sample(_index_select(_index_select(gamma, j - 1) , k - 1) , "gamma[%d][%d]" % (to_int(j-1),to_int(k-1)), "normal", [0, 320])
    for i in range(1, to_int(I) + 1):
        for j in range(1, to_int(J) + 1):

            lambda_[i - 1][j - 1] =  _pyro_sample(_index_select(_index_select(lambda_, i - 1) , j - 1) , "lambda_[%d][%d]" % (to_int(i-1),to_int(j-1)), "normal", [0, 320])
            X[i - 1][j - 1] =  _pyro_sample(_index_select(_index_select(X, i - 1) , j - 1) , "X[%d][%d]" % (to_int(i-1),to_int(j-1)), "poisson_log", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(_index_select(lambda_, i - 1) , j - 1) ,alpha]),_index_select(beta, i - 1) ]),_index_select(gamma, j - 1) ])], obs=_index_select(_index_select(X, i - 1) , j - 1) )

