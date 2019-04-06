# model file: example-models/ARM/Ch.7/congress.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'incumbency_88' in data, 'variable not found in data: key=incumbency_88'
    assert 'vote_86' in data, 'variable not found in data: key=vote_86'
    assert 'vote_88' in data, 'variable not found in data: key=vote_88'
    # initialize data
    N = data["N"]
    incumbency_88 = data["incumbency_88"]
    vote_86 = data["vote_86"]
    vote_88 = data["vote_88"]

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(3)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    incumbency_88 = data["incumbency_88"]
    vote_86 = data["vote_86"]
    vote_88 = data["vote_88"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    sigma =  pyro.sample("sigma", dist.HalfCauchy(torch.tensor(2.5)))
    with pyro.plate("data", N):
        vote_88 = pyro.sample('vote_88', dist.Normal(beta[...,0] + beta[...,1] * vote_86 + beta[...,2] * incumbency_88, sigma), obs=vote_88)

