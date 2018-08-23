# model file: ../example-models/bugs_examples/vol2/dugongs/dugongs.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'Y' in data, 'variable not found in data: key=Y'
    # initialize data
    N = data["N"]
    x = data["x"]
    Y = data["Y"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(x, dims=[N])
    check_constraints(Y, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    Y = data["Y"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha") # real/double
    params["beta"] = init_real("beta") # real/double
    params["lambda_"] = init_real("lambda_", low=0.5, high=1) # real/double
    params["tau"] = init_real("tau", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    Y = data["Y"]
    # INIT parameters
    alpha = params["alpha"]
    beta = params["beta"]
    lambda_ = params["lambda_"]
    tau = params["tau"]
    # initialize transformed parameters
    sigma = init_real("sigma") # real/double
    U3 = init_real("U3") # real/double
    sigma = _pyro_assign(sigma, (1 / _call_func("sqrt", [tau])))
    U3 = _pyro_assign(U3, _call_func("logit", [lambda_]))
    # model block
    # {
    m = init_real("m", dims=(N)) # real/double

    for i in range(1, to_int(N) + 1):
        m[i - 1] = _pyro_assign(m[i - 1], (alpha - (beta * _call_func("pow", [lambda_,_index_select(x, i - 1) ]))))
    Y =  _pyro_sample(Y, "Y", "normal", [m, sigma], obs=Y)
    alpha =  _pyro_sample(alpha, "alpha", "normal", [0.0, 1000])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0, 1000])
    lambda_ =  _pyro_sample(lambda_, "lambda_", "uniform", [0.5, 1])
    tau =  _pyro_sample(tau, "tau", "gamma", [0.0001, 0.0001])
    # }

