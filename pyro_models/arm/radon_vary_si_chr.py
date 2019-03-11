# model file: example-models/ARM/Ch.13/radon_vary_si_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'county' in data, 'variable not found in data: key=county'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    county = data["county"]
    x = data["x"]
    y = data["y"]

def init_params(data):
    params = {}
    params["sigma_a1"] = pyro.sample("sigma_a1", dist.Uniform(0., 100.))
    params["sigma_a2"] = pyro.sample("sigma_a2", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    county = data["county"].long() - 1
    x = data["x"]
    y = data["y"]

    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_y = params["sigma_y"]

    mu_a1 =  pyro.sample("mu_a1", dist.Normal(0., 1.))
    mu_a2 =  pyro.sample("mu_a2", dist.Normal(0., 1.))

    with pyro.plate("mu", 85):
        eta1 =  pyro.sample("eta1", dist.Normal(0., 1.))
        eta2 =  pyro.sample("eta2", dist.Normal(0., 1.))

    a1 = mu_a1 + sigma_a1 * eta1
    a2 = 0.1 * mu_a2 + sigma_a2 * eta2

    with pyro.plate("data", N):
        y_hat = a1[county] + a2[county] * x
        y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)

