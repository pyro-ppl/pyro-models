# model file: example-models/ARM/Ch.5/wells_d100ars.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'arsenic' in data, 'variable not found in data: key=arsenic'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist_ = data["dist"]
    arsenic = data["arsenic"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist_ = data["dist"]
    arsenic = data["arsenic"]
    dist100 = dist_ / 100.
    data["dist100"] = dist100

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist_ = data["dist"]
    arsenic = data["arsenic"]
    # initialize transformed data
    dist100 = data["dist100"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block
    with pyro.plate("data", N):
        switched = pyro.sample('switched', dist.Bernoulli(logits=beta[0] + beta[1] * dist100 + beta[2] * arsenic), obs=switched)

