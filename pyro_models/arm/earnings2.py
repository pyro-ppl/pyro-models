# model file: example-models/ARM/Ch.7/earnings2.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earnings' in data, 'variable not found in data: key=earnings'
    assert 'height' in data, 'variable not found in data: key=height'
    assert 'sex' in data, 'variable not found in data: key=sex'
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]
    log_earnings = torch.log(earnings)
    male = 2 - sex
    data["log_earnings"] = log_earnings
    data["male"] = male

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(3)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]
    # initialize transformed data
    log_earnings = data["log_earnings"]
    male = data["male"]

    # init parameters
    beta = params["beta"]

    sigma =  pyro.sample("sigma", dist.HalfCauchy(torch.tensor(2.5)))
    with pyro.plate("data", N):    
        log_earnings = pyro.sample('log_earnings', dist.Normal(beta[...,0] + beta[...,1] * height + beta[...,2] * male, sigma), obs=log_earnings)
