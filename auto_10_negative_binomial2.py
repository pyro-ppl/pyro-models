# model file: ../example-models/basic_estimators/negative_binomial2.stan
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
    check_constraints(N, low=1, dims=[1])
    check_constraints(y, low=0, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha", low=0) # real/double
    params["p_success"] = init_real("p_success", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # INIT parameters
    alpha = params["alpha"]
    p_success = params["p_success"]
    # initialize transformed parameters
    beta = init_real("beta", low=0.0) # real/double
    beta = _pyro_assign(beta, (p_success / (1.0 - p_success)))
    # model block

    for i in range(1, to_int(N) + 1):
        y[i - 1] =  _pyro_sample(_index_select(y, i - 1) , "y[%d]" % (to_int(i-1)), "neg_binomial", [alpha, beta], obs=_index_select(y, i - 1) )

