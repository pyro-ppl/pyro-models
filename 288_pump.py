# model file: ../example-models/bugs_examples/vol1/pump/pump.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 't' in data, 'variable not found in data: key=t'
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]
    # assign init values for parameters
    params["alpha"] = pyro.sample("alpha", dist.Uniform(0))
    params["beta"] = pyro.sample("beta", dist.Uniform(0))
    params["theta"] = init_vector("theta", dist.Uniform(0., dims=(N)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]
    
    # init parameters
    alpha = params["alpha"]
    beta = params["beta"]
    theta = params["theta"]
    # initialize transformed parameters
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "exponential", [1.0])
    beta =  _pyro_sample(beta, "beta", "gamma", [0.10000000000000001, 1.0])
    theta =  _pyro_sample(theta, "theta", "gamma", [alpha, beta])
    x =  _pyro_sample(x, "x", "poisson", [_call_func("elt_multiply", [theta,t])], obs=x)

