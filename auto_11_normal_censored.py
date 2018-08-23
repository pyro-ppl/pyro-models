# model file: ../example-models/basic_estimators/normal_censored.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'U' in data, 'variable not found in data: key=U'
    assert 'N_censored' in data, 'variable not found in data: key=N_censored'
    assert 'N_observed' in data, 'variable not found in data: key=N_observed'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    U = data["U"]
    N_censored = data["N_censored"]
    N_observed = data["N_observed"]
    y = data["y"]
    check_constraints(U, dims=[1])
    check_constraints(N_censored, low=0, dims=[1])
    check_constraints(N_observed, low=0, dims=[1])
    check_constraints(y, high=U, dims=[N_observed])

def init_params(data, params):
    # initialize data
    U = data["U"]
    N_censored = data["N_censored"]
    N_observed = data["N_observed"]
    y = data["y"]
    # assign init values for parameters
    params["mu"] = init_real("mu") # real/double

def model(data, params):
    # initialize data
    U = data["U"]
    N_censored = data["N_censored"]
    N_observed = data["N_observed"]
    y = data["y"]
    # INIT parameters
    mu = params["mu"]
    # initialize transformed parameters
    # model block

    for n in range(1, to_int(N_observed) + 1):
        y[n - 1] =  _pyro_sample(_index_select(y, n - 1) , "y[%d]" % (to_int(n-1)), "normal", [mu, 1.0], obs=_index_select(y, n - 1) )
    pyro.sample("(N_censored * _call_func( log1m , [_call_func( normal_cdf , [U,mu,1.0])]))", dist.Bernoulli((N_censored * _call_func("log1m", [_call_func("normal_cdf", [U,mu,1.0])]))), obs=(1));

