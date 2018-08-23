# model file: ../example-models/bugs_examples/vol2/beetles/beetles_logit.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'r' in data, 'variable not found in data: key=r'
    assert 'x' in data, 'variable not found in data: key=x'
    # initialize data
    N = data["N"]
    n = data["n"]
    r = data["r"]
    x = data["x"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(n, low=0, dims=[N])
    check_constraints(r, low=0, dims=[N])
    check_constraints(x, dims=[N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    n = data["n"]
    r = data["r"]
    x = data["x"]
    centered_x = init_vector("centered_x", dims=(N)) # vector
    mean_x = init_real("mean_x") # real/double
    mean_x = _pyro_assign(mean_x, _call_func("mean", [x]))
    centered_x = _pyro_assign(centered_x, _call_func("subtract", [x,mean_x]))
    data["centered_x"] = centered_x
    data["mean_x"] = mean_x

def init_params(data, params):
    # initialize data
    N = data["N"]
    n = data["n"]
    r = data["r"]
    x = data["x"]
    # initialize transformed data
    centered_x = data["centered_x"]
    mean_x = data["mean_x"]
    # assign init values for parameters
    params["alpha_star"] = init_real("alpha_star") # real/double
    params["beta"] = init_real("beta") # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n = data["n"]
    r = data["r"]
    x = data["x"]
    # initialize transformed data
    centered_x = data["centered_x"]
    mean_x = data["mean_x"]
    # INIT parameters
    alpha_star = params["alpha_star"]
    beta = params["beta"]
    # initialize transformed parameters
    m = init_vector("m", dims=(N)) # vector
    m = _pyro_assign(m, _call_func("add", [alpha_star,_call_func("multiply", [beta,centered_x])]))
    # model block

    alpha_star =  _pyro_sample(alpha_star, "alpha_star", "normal", [0.0, 10000.0])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0, 10000.0])
    r =  _pyro_sample(r, "r", "binomial_logit", [n, m], obs=r)

