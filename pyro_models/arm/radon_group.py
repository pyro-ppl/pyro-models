# model file: example-models/ARM/Ch.12/radon_group.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'county' in data, 'variable not found in data: key=county'
    assert 'u' in data, 'variable not found in data: key=u'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    J = data["J"]
    county = data["county"]
    u = data["u"]
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
    u = data["u"]
    x = data["x"]
    y = data["y"]

    sigma =  pyro.sample("sigma", dist.HalfCauchy(2.5))
    mu_alpha =  pyro.sample("mu_alpha", dist.Normal(0., 1.))
    sigma_alpha =  pyro.sample("sigma_alpha", dist.HalfCauchy(2.5))
    mu_beta =  pyro.sample("mu_beta", dist.Normal(0., 1.))
    sigma_beta =  pyro.sample("sigma_beta", dist.HalfCauchy(2.5))
    with pyro.plate("J", J):
        alpha =  pyro.sample("alpha", dist.Normal(mu_alpha, sigma_alpha))
    with pyro.plate("2", 2):
        beta =  pyro.sample("beta", dist.Normal(mu_beta, sigma_beta))
    with pyro.plate("data", N):
        y_hat = alpha[...,county] + x * beta[...,0].unsqueeze(-1) + u * beta[...,1].unsqueeze(-1)
        y = pyro.sample('y', dist.Normal(y_hat, sigma), obs=y)

