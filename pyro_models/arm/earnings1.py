# model file: example-models/ARM/Ch.7/earnings1.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn_pos' in data, 'variable not found in data: key=earn_pos'
    assert 'height' in data, 'variable not found in data: key=height'
    assert 'male' in data, 'variable not found in data: key=male'
    # initialize data
    N = data["N"]
    earn_pos = data["earn_pos"]
    height = data["height"]
    male = data["male"]

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(3)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    earn_pos = data["earn_pos"]
    height = data["height"]
    male = data["male"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    sigma =  pyro.sample("sigma", dist.HalfCauchy(torch.tensor(2.5)))
    with pyro.plate("data", N):    
        earn_pos = pyro.sample('earn_pos', dist.Bernoulli(logits=beta[...,0] + beta[...,1] * height + beta[...,2] * male), obs=earn_pos)
