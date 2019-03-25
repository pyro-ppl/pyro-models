# model file: example-models/ARM/Ch.5/wells_dist.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist_ = data["dist"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block
    with pyro.plate("data", N):
        log_p = beta[...,0] + beta[...,1] * dist_
        switched_sample = pyro.sample('switched', dist.Bernoulli(logits=log_p), obs=switched)
