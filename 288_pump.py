# model file: ../example-models/bugs_examples/vol1/pump/pump.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 't' in data, 'variable not found in data: key=t'
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha", low=0) # real/double
    params["beta"] = init_real("beta", low=0) # real/double
    params["theta"] = init_vector("theta", low=0, dims=(N)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]
    # INIT parameters
    alpha = params["alpha"]
    beta = params["beta"]
    theta = params["theta"]
    # initialize transformed parameters
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "exponential", [1.0])
    beta =  _pyro_sample(beta, "beta", "gamma", [0.10000000000000001, 1.0])
    theta =  _pyro_sample(theta, "theta", "gamma", [alpha, beta])
    x =  _pyro_sample(x, "x", "poisson", [_call_func("elt_multiply", [theta,t])], obs=x)

