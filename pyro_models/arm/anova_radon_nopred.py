# model file: example-models/ARM/Ch.22/anova_radon_nopred.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'county' in data, 'variable not found in data: key=county'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    N = data["N"]
    county = data["county"]
    y = data["y"]

def init_params(data):
    params = {}
    # assign init values for parameters
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    J = data["J"]
    N = data["N"]
    county = data["county"].long() - 1
    y = data["y"]

    # init parameters
    sigma_a = params["sigma_a"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters

    mu_a =  pyro.sample("mu_a", dist.Normal(0., 1))
    with pyro.plate("J", J):
        a =  pyro.sample("a", dist.Normal((10 * mu_a), sigma_a))
    with pyro.plate("data", N):
        y_hat = a[...,county]
        y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)

