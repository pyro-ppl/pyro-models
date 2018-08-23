# model file: ../example-models/bugs_examples/vol1/seeds/seeds.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'I' in data, 'variable not found in data: key=I'
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x1' in data, 'variable not found in data: key=x1'
    assert 'x2' in data, 'variable not found in data: key=x2'
    # initialize data
    I = data["I"]
    n = data["n"]
    N = data["N"]
    x1 = data["x1"]
    x2 = data["x2"]
    check_constraints(I, low=0, dims=[1])
    check_constraints(n, low=0, dims=[I])
    check_constraints(N, low=0, dims=[I])
    check_constraints(x1, dims=[I])
    check_constraints(x2, dims=[I])

def transformed_data(data):
    # initialize data
    I = data["I"]
    n = data["n"]
    N = data["N"]
    x1 = data["x1"]
    x2 = data["x2"]
    x1x2 = init_vector("x1x2", dims=(I)) # vector
    x1x2 = _pyro_assign(x1x2, _call_func("elt_multiply", [x1,x2]))
    data["x1x2"] = x1x2

def init_params(data, params):
    # initialize data
    I = data["I"]
    n = data["n"]
    N = data["N"]
    x1 = data["x1"]
    x2 = data["x2"]
    # initialize transformed data
    x1x2 = data["x1x2"]
    # assign init values for parameters
    params["alpha0"] = init_real("alpha0") # real/double
    params["alpha1"] = init_real("alpha1") # real/double
    params["alpha12"] = init_real("alpha12") # real/double
    params["alpha2"] = init_real("alpha2") # real/double
    params["tau"] = init_real("tau", low=0) # real/double
    params["b"] = init_vector("b", dims=(I)) # vector

def model(data, params):
    # initialize data
    I = data["I"]
    n = data["n"]
    N = data["N"]
    x1 = data["x1"]
    x2 = data["x2"]
    # initialize transformed data
    x1x2 = data["x1x2"]
    # INIT parameters
    alpha0 = params["alpha0"]
    alpha1 = params["alpha1"]
    alpha12 = params["alpha12"]
    alpha2 = params["alpha2"]
    tau = params["tau"]
    b = params["b"]
    # initialize transformed parameters
    sigma = init_real("sigma", low=0) # real/double
    sigma = _pyro_assign(sigma, (1.0 / _call_func("sqrt", [tau])))
    # model block

    alpha0 =  _pyro_sample(alpha0, "alpha0", "normal", [0.0, 1000.0])
    alpha1 =  _pyro_sample(alpha1, "alpha1", "normal", [0.0, 1000.0])
    alpha2 =  _pyro_sample(alpha2, "alpha2", "normal", [0.0, 1000.0])
    alpha12 =  _pyro_sample(alpha12, "alpha12", "normal", [0.0, 1000.0])
    tau =  _pyro_sample(tau, "tau", "gamma", [0.001, 0.001])
    b =  _pyro_sample(b, "b", "normal", [0.0, sigma])
    n =  _pyro_sample(n, "n", "binomial_logit", [N, _call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [alpha0,_call_func("multiply", [alpha1,x1])]),_call_func("multiply", [alpha2,x2])]),_call_func("multiply", [alpha12,x1x2])]),b])], obs=n)

