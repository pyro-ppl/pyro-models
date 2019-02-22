# model file: example-models/ARM/Ch.7/wells.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'switc' in data, 'variable not found in data: key=switc'
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    dist_ = data["dist"]
    switc = data["switc"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block
    with pyro.plate("data", N):
        switc= pyro.sample('switc', dist.Bernoulli(logits=beta[0] + beta[1] * dist_), obs=switc)

