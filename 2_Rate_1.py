# model file: ../example-models/Bayesian_Cognitive_Modeling/GettingStarted/Rate_1.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))


def validate_data_def(data):
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'k' in data, 'variable not found in data: key=k'
    # initialize data
    n = data["n"]
    k = data["k"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    n = torch.tensor(data["n"]).float()
    k = torch.tensor(data["k"]).float()

    theta =  pyro.sample("theta", dist.Beta(1., 1.))
    pyro.sample("k", dist.Binomial(n, theta), obs=k)

