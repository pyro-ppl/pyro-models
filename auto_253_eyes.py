# model file: ../example-models/bugs_examples/vol2/eyes/eyes.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    y = data["y"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(y, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["sigmasq"] = init_real("sigmasq", low=0) # real/double
    params["theta"] = init_real("theta", low=0) # real/double
    params["lambda_1"] = init_real("lambda_1") # real/double
    params["p1"] = init_real("p1", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # INIT parameters
    sigmasq = params["sigmasq"]
    theta = params["theta"]
    lambda_1 = params["lambda_1"]
    p1 = params["p1"]
    # initialize transformed parameters
    lambda_ = init_real("lambda_", dims=(2)) # real/double
    sigma = init_real("sigma") # real/double
    sigma = _pyro_assign(sigma, _call_func("sqrt", [sigmasq]))
    lambda_[1 - 1] = _pyro_assign(lambda_[1 - 1], lambda_1)
    lambda_[2 - 1] = _pyro_assign(lambda_[2 - 1], (_index_select(lambda_, 1 - 1)  + theta))
    # model block

    theta =  _pyro_sample(theta, "theta", "normal", [0, 100])
    lambda_1 =  _pyro_sample(lambda_1, "lambda_1", "normal", [0, 1000.0])
    sigmasq =  _pyro_sample(sigmasq, "sigmasq", "inv_gamma", [0.001, 0.001])
    # {
    log_p1 = init_real("log_p1") # real/double
    log1m_p1 = init_real("log1m_p1") # real/double

    log_p1 = _pyro_assign(log_p1, _call_func("log", [p1]))
    log1m_p1 = _pyro_assign(log1m_p1, _call_func("log1m", [p1]))
    for n in range(1, to_int(N) + 1):
        pyro.sample("_call_func( log_sum_exp , [(log_p1 + _call_func( normal_log , [_index_select(y, n - 1) ,_index_select(lambda_, 1 - 1) ,sigma])),(log1m_p1 + _call_func( normal_log , [_index_select(y, n - 1) ,_index_select(lambda_, 2 - 1) ,sigma]))])[%d]" % (n), dist.Bernoulli(_call_func("log_sum_exp", [(log_p1 + _call_func("normal_log", [_index_select(y, n - 1) ,_index_select(lambda_, 1 - 1) ,sigma])),(log1m_p1 + _call_func("normal_log", [_index_select(y, n - 1) ,_index_select(lambda_, 2 - 1) ,sigma]))])), obs=(1));
    # }

