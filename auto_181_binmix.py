# model file: ../example-models/BPA/Ch.12/binmix.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'R' in data, 'variable not found in data: key=R'
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'K' in data, 'variable not found in data: key=K'
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    K = data["K"]
    check_constraints(R, low=0, dims=[1])
    check_constraints(T, low=0, dims=[1])
    check_constraints(y, low=0, dims=[R, T])
    check_constraints(K, low=0, dims=[1])

def transformed_data(data):
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    K = data["K"]
    max_y = init_int("max_y", low=0, dims=(R)) # real/double
    for i in range(1, to_int(R) + 1):
        max_y[i - 1] = _pyro_assign(max_y[i - 1], _call_func("max", [_index_select(y, i - 1) ]))
    data["max_y"] = max_y

def init_params(data, params):
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    K = data["K"]
    # initialize transformed data
    max_y = data["max_y"]
    # assign init values for parameters
    params["lambda_"] = init_real("lambda_", low=0) # real/double
    params["p"] = init_real("p", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    K = data["K"]
    # initialize transformed data
    max_y = data["max_y"]
    # INIT parameters
    lambda_ = params["lambda_"]
    p = params["p"]
    # initialize transformed parameters
    # model block

    lambda_ =  _pyro_sample(lambda_, "lambda_", "cauchy", [0, 10])
    for i in range(1, to_int(R) + 1):
        # {
        lp = init_vector("lp", dims=(((K - max_y[i - 1]) + 1))) # vector

        for j in range(1, to_int(((K - max_y[i - 1]) + 1)) + 1):
            lp[j - 1] = _pyro_assign(lp[j - 1], (_call_func("poisson_log", [((_index_select(max_y, i - 1)  + j) - 1),lambda_]) + _call_func("binomial_log", [_index_select(y, i - 1) ,((_index_select(max_y, i - 1)  + j) - 1),p])))
        pyro.sample("_call_func( log_sum_exp , [lp])[%d]" % (i), dist.Bernoulli(_call_func("log_sum_exp", [lp])), obs=(1));
        # }

