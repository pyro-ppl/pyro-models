# model file: ../example-models/misc/irt/irt.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'jj' in data, 'variable not found in data: key=jj'
    assert 'kk' in data, 'variable not found in data: key=kk'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"]
    kk = data["kk"]
    y = data["y"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"].long() - 1
    kk = data["kk"].long() - 1
    y = data["y"]

    # initialize transformed parameters
    # model block

    with pyro.plate('alpha_', J):
        alpha =  pyro.sample("alpha", dist.Normal(0., 1.))
    with pyro.plate('beta', K):
        beta =  pyro.sample("beta_", dist.Normal(0., 1.))
    delta =  pyro.sample("delta", dist.Normal(0.75, 1.))
    with pyro.plate('data', N):
        y = pyro.sample('y', dist.Bernoulli(logits=alpha[jj] - beta[kk] + delta), obs=y)

