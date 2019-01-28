# model file: ../example-models/ARM/Ch.5/wells_dist100.stan
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

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    dist100 = dist / 100.
    data["dist100"] = dist100

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    # initialize transformed data
    dist100 = data["dist100"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    pyro.sample('switched', dist.Bernoulli(logits=beta[0] + beta[1] * dist100), obs=switched)
