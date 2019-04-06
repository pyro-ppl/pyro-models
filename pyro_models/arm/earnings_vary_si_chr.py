# model file: example-models/ARM/Ch.13/earnings_vary_si_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'eth' in data, 'variable not found in data: key=eth'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    log_earn = torch.log(earn)
    data["log_earn"] = log_earn

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"].long() - 1
    height = data["height"]
    # initialize transformed data
    log_earn = data["log_earn"]

    mu_a1 = pyro.sample('mu_a1', dist.Normal(0., 1.))
    mu_a2 = pyro.sample('mu_a2', dist.Normal(0., 1.))
    sigma_a1 =  pyro.sample("sigma_a1", dist.HalfCauchy(5.))
    sigma_a2 =  pyro.sample("sigma_a2", dist.HalfCauchy(5.))
    sigma_y =  pyro.sample("sigma_y", dist.HalfCauchy(5.))
    with pyro.plate('etas', 4):
        eta1 = pyro.sample('eta1', dist.Normal(0., 1.))
        eta2 = pyro.sample('eta2', dist.Normal(0., 1.))
    a1 = 10 * mu_a1 + sigma_a1 * eta1
    a2 = 0.1 * mu_a2 + sigma_a2 * eta2
    with pyro.plate('data', N):
        y_hat = a1[...,eth] + a2[...,eth] * height
        log_earn = pyro.sample("log_earn", dist.Normal(y_hat, sigma_y), obs=log_earn)
