# model file: example-models/ARM/Ch.12/radon_complete_pool.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    x = data["x"]
    y = data["y"]

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    y = data["y"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    sigma =  pyro.sample("sigma", dist.HalfCauchy(2.5))
    with pyro.plate("data", N):
        y = pyro.sample('y', dist.Normal(beta[...,0] + beta[...,1] * x, sigma), obs=y)
