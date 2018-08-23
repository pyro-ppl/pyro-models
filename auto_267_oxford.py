# model file: ../example-models/bugs_examples/vol1/oxford/oxford.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'n0' in data, 'variable not found in data: key=n0'
    assert 'n1' in data, 'variable not found in data: key=n1'
    assert 'r0' in data, 'variable not found in data: key=r0'
    assert 'r1' in data, 'variable not found in data: key=r1'
    assert 'year' in data, 'variable not found in data: key=year'
    # initialize data
    K = data["K"]
    n0 = data["n0"]
    n1 = data["n1"]
    r0 = data["r0"]
    r1 = data["r1"]
    year = data["year"]
    check_constraints(K, low=0, dims=[1])
    check_constraints(n0, low=0, dims=[K])
    check_constraints(n1, low=0, dims=[K])
    check_constraints(r0, low=0, dims=[K])
    check_constraints(r1, low=0, dims=[K])
    check_constraints(year, dims=[K])

def transformed_data(data):
    # initialize data
    K = data["K"]
    n0 = data["n0"]
    n1 = data["n1"]
    r0 = data["r0"]
    r1 = data["r1"]
    year = data["year"]
    yearsq = init_vector("yearsq", dims=(K)) # vector
    yearsq = _pyro_assign(yearsq, _call_func("elt_multiply", [year,year]))
    data["yearsq"] = yearsq

def init_params(data, params):
    # initialize data
    K = data["K"]
    n0 = data["n0"]
    n1 = data["n1"]
    r0 = data["r0"]
    r1 = data["r1"]
    year = data["year"]
    # initialize transformed data
    yearsq = data["yearsq"]
    # assign init values for parameters
    params["mu"] = init_vector("mu", dims=(K)) # vector
    params["alpha"] = init_real("alpha") # real/double
    params["beta1"] = init_real("beta1") # real/double
    params["beta2"] = init_real("beta2") # real/double
    params["sigma_sq"] = init_real("sigma_sq", low=0) # real/double
    params["b"] = init_vector("b", dims=(K)) # vector

def model(data, params):
    # initialize data
    K = data["K"]
    n0 = data["n0"]
    n1 = data["n1"]
    r0 = data["r0"]
    r1 = data["r1"]
    year = data["year"]
    # initialize transformed data
    yearsq = data["yearsq"]
    # INIT parameters
    mu = params["mu"]
    alpha = params["alpha"]
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    sigma_sq = params["sigma_sq"]
    b = params["b"]
    # initialize transformed parameters
    sigma = init_real("sigma", low=0) # real/double
    sigma = _pyro_assign(sigma, _call_func("sqrt", [sigma_sq]))
    # model block

    r0 =  _pyro_sample(r0, "r0", "binomial_logit", [n0, mu], obs=r0)
    r1 =  _pyro_sample(r1, "r1", "binomial_logit", [n1, _call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [alpha,mu]),_call_func("multiply", [beta1,year])]),_call_func("multiply", [beta2,_call_func("subtract", [yearsq,22])])]),_call_func("multiply", [b,sigma])])], obs=r1)
    b =  _pyro_sample(b, "b", "normal", [0, 1])
    mu =  _pyro_sample(mu, "mu", "normal", [0, 1000])
    alpha =  _pyro_sample(alpha, "alpha", "normal", [0.0, 1000])
    beta1 =  _pyro_sample(beta1, "beta1", "normal", [0.0, 1000])
    beta2 =  _pyro_sample(beta2, "beta2", "normal", [0.0, 1000])
    sigma_sq =  _pyro_sample(sigma_sq, "sigma_sq", "inv_gamma", [0.001, 0.001])

