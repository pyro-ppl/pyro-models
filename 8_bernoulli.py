# model file: ../example-models/basic_estimators/bernoulli.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    y = data["y"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    theta =  pyro.sample("theta", dist.Beta(1., 1.))
    pyro.sample('obs', dist.Bernoulli(theta), obs=y)

