# model file: ../example-models/ARM/Ch.10/sesame_one_pred_2b.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'watched_hat' in data, 'variable not found in data: key=watched_hat'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    watched_hat = data["watched_hat"]
    y = data["y"]

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(2)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    watched_hat = data["watched_hat"]
    y = data["y"]

    # init parameters
    beta = params["beta"]
    sigma = pyro.sample('sigma', dist.HalfCauchy(2.5))
    # initialize transformed parameters
    # model block

    with pyro.plate('data', N):
        y = pyro.sample('y', dist.Normal(beta[0] + beta[1] * watched_hat, sigma), obs=y)

