# model file: example-models/ARM/Ch.5/wells_dae.stan
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
    assert 'educ' in data, 'variable not found in data: key=educ'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist_ = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist_ = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    dist100 = dist_ / 100.0;
    educ4   = educ / 4.0;
    data["dist100"] = dist100
    data["educ4"] = educ4

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist_ = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    # initialize transformed data
    dist100 = data["dist100"]
    educ4 = data["educ4"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    with pyro.plate("data", N):
        switched = pyro.sample('switched', dist.Bernoulli(logits=beta[0] + beta[1] * dist100 + \
                        beta[2] * arsenic + beta[3] * educ4), obs=switched)

