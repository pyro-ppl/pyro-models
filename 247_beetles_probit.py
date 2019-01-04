# model file: ../example-models/bugs_examples/vol2/beetles/beetles_probit.stan
import torch
import pyro


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
    p = init_real("p", dims=(N)) # real/double
    for i in range(1, to_int(N) + 1):
        p[i - 1] = _pyro_assign(p[i - 1], _call_func("Phi", [(alpha_star + (beta * _index_select(centered_x, i - 1) ))]))
    # model block

    alpha_star =  _pyro_sample(alpha_star, "alpha_star", "normal", [0.0, 1.0])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0, 10000.0])
    r =  _pyro_sample(r, "r", "binomial", [n, p], obs=r)

