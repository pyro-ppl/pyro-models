# model file: ../example-models/bugs_examples/vol1/litter/litter.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'G' in data, 'variable not found in data: key=G'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'r' in data, 'variable not found in data: key=r'
    assert 'n' in data, 'variable not found in data: key=n'
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # XXX: this model currenty NaNs
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]

    # model block
    with pyro.plate('a_', G, dim=-2):
        mu = pyro.sample('mu', dist.Uniform(0., 1.))
        a_plus_b = pyro.sample('a_plus_b', dist.Pareto(0.1, 1.5))
        a = mu * a_plus_b
        b = (1 - mu) * a_plus_b
        with pyro.plate('data', N, dim=-1):
            p = pyro.sample('p', dist.Beta(a, b))
            r = pyro.sample('r', dist.Binomial(n, p), obs=r)

