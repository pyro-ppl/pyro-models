# model file: ../example-models/ARM/Ch.12/radon_group_chr.stan
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

    mu_b =  pyro.sample("mu_b", dist.Normal(0., 1.).expand([J]))
    eta =  pyro.sample("eta", dist.Normal(0., 1.).expand([J]))
    beta =  pyro.sample("beta", dist.Normal(0., 100.).expand([2]))
    sigma =  pyro.sample("sigma", dist.HalfCauchy(2.5))
    sigma_b =  pyro.sample("sigma_b", dist.HalfCauchy(2.5).expand([J]))
    b = mu_b + sigma_b * eta
    y_hat = b[county] + x * beta[0] + u * beta[1]
    y = pyro.sample('y', dist.Normal(y_hat, sigma), obs=y)
