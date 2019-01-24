# model file: ../example-models/ARM/Ch.19/radon_redundant_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



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
    return params

def model(data, params):
    # initialize data
    J = data["J"]
    N = data["N"]
    county = data["county"].long() - 1
    y = data["y"]

    mu_eta =  pyro.sample("mu_eta", dist.Normal(0., 1.))
    sigma_eta =  pyro.sample("sigma_eta", dist.Uniform(0., 100.))
    sigma_y =  pyro.sample("sigma_y", dist.Uniform(0., 100.))
    et =  pyro.sample("eta", dist.Normal(0., sigma_eta).expand([J]))
    eta = 100 * mu_eta + sigma_eta * et
    y_hat = 100 * mu_eta + eta[county]
    y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)
