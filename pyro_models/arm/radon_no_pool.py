# model file: example-models/ARM/Ch.12/radon_no_pool.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'county' in data, 'variable not found in data: key=county'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    J = data["J"]
    county = data["county"]
    x = data["x"]
    y = data["y"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    J = data["J"]
    county = data["county"].long() - 1
    x = data["x"]
    y = data["y"]

    # model block
    mu_a =  pyro.sample("mu_a", dist.Normal(0., 1.))
    sigma_a =  pyro.sample("sigma_a", dist.HalfCauchy(2.5))
    with pyro.plate("J", J):    
        a = pyro.sample("a", dist.Normal(mu_a, sigma_a))
    beta =  pyro.sample("beta", dist.Normal(0., 1.))
    sigma_y =  pyro.sample("sigma_y", dist.HalfCauchy(2.5))
    with pyro.plate("data", N):
        y_hat = beta * x + a[county]
        y = pyro.sample('y', dist.Normal(y_hat, sigma_y), obs=y)
