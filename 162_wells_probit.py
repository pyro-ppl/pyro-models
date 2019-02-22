# model file: example-models/ARM/Ch.6/wells_probit.stan
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

def transformed_data(data):
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    dist100 = dist / 100.
    data["dist100"] = dist100

def init_params(data):
    params = {}
    dist100 = data["dist100"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    switc = data["switc"]
    # initialize transformed data
    dist100 = data["dist100"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    with pyro.plate("data", N):
        phi = dist.Normal(0., 1.).cdf
        switc = pyro.sample('switched', dist.Bernoulli(logits=phi(beta[0] + beta[1] * dist100)), obs=switc)
