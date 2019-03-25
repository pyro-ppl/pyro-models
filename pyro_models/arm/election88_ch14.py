# model file: example-models/ARM/Ch.14/election88.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_state' in data, 'variable not found in data: key=n_state'
    assert 'black' in data, 'variable not found in data: key=black'
    assert 'female' in data, 'variable not found in data: key=female'
    assert 'state' in data, 'variable not found in data: key=state'
    assert 'y' in data, 'variable not found in data: key=y'

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_state = data["n_state"]
    black = data["black"]
    female = data["female"]
    state = data["state"].long() - 1
    y = data["y"]

    # model block
    mu_a =  pyro.sample("mu_a", dist.Normal(0., 1.))
    sigma_a =  pyro.sample("sigma_a", dist.Uniform(0., 100.))

    with pyro.plate('a_plate', n_state):
        a = pyro.sample("a", dist.Normal(mu_a, sigma_a))

    with pyro.plate('b_plate', 2):
        b = pyro.sample("b", dist.Normal(0., 100.))

    with pyro.plate('data', N):
        y_hat = b[...,0].unsqueeze(-1)*black + b[...,1].unsqueeze(-1)*female + a[...,state]
        y =  pyro.sample("y", dist.Bernoulli(logits=y_hat), obs=y)
