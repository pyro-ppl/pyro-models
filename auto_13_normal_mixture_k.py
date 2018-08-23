# model file: ../example-models/basic_estimators/normal_mixture_k.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    K = data["K"]
    N = data["N"]
    y = data["y"]
    check_constraints(K, low=1, dims=[1])
    check_constraints(N, low=1, dims=[1])
    check_constraints(y, dims=[N])

def init_params(data, params):
    # initialize data
    K = data["K"]
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["theta"] = init_simplex("theta") # real/double
    params["mu"] = init_real("mu", dims=(K)) # real/double
    params["sigma"] = init_real("sigma", low=0, high=10, dims=(K)) # real/double

def model(data, params):
    # initialize data
    K = data["K"]
    N = data["N"]
    y = data["y"]
    # INIT parameters
    theta = params["theta"]
    mu = params["mu"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block
    # {
    ps = init_real("ps", dims=(K)) # real/double

    mu =  _pyro_sample(mu, "mu", "normal", [0, 10])
    for n in range(1, to_int(N) + 1):

        for k in range(1, to_int(K) + 1):
            ps[k - 1] = _pyro_assign(ps[k - 1], (_call_func("log", [_index_select(theta, k - 1) ]) + _call_func("normal_log", [_index_select(y, n - 1) ,_index_select(mu, k - 1) ,_index_select(sigma, k - 1) ])))
        pyro.sample("_call_func( log_sum_exp , [ps])[%d]" % (n), dist.Bernoulli(_call_func("log_sum_exp", [ps])), obs=(1));
    # }

