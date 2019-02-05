# model file: ../example-models/ARM/Ch.10/sesame_one_pred_a.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'encouraged' in data, 'variable not found in data: key=encouraged'
    assert 'watched' in data, 'variable not found in data: key=watched'
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(2)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]

    # init parameters
    beta = params["beta"]
    sigma = pyro.sample('sigma', dist.HalfCauchy(2.5))
    # initialize transformed parameters
    with pyro.plate('data', N):
        watched = pyro.sample('y', dist.Normal(beta[0] + beta[1] * encouraged, sigma), obs=watched)

