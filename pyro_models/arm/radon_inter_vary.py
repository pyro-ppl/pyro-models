# model file: example-models/ARM/Ch.13/radon_inter_vary.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'county' in data, 'variable not found in data: key=county'
    assert 'u' in data, 'variable not found in data: key=u'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    county = data["county"]
    u = data["u"]
    x = data["x"]
    y = data["y"]

def init_params(data):
    params = {}
    # assign init values for parameters
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    params["sigma_b"] = pyro.sample("sigma_b", dist.Uniform(0., 100.))
    params["sigma_beta"] = pyro.sample("sigma_beta", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    county = data["county"].long() - 1
    u = data["u"]
    x = data["x"]
    y = data["y"]
    # initialize transformed data
    inter = u * x

    # init parameters
    sigma_a = params["sigma_a"]
    sigma_b = params["sigma_b"]
    sigma_beta = params["sigma_beta"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    mu_beta =  pyro.sample("mu_beta", dist.Normal(0., 1.))
    with pyro.plate("beta_plate", 2):
        beta =  pyro.sample("beta", dist.Normal(100 * mu_beta, sigma_beta))
    mu_a =  pyro.sample("mu_a", dist.Normal(0., 1.))
    mu_b =  pyro.sample("mu_b", dist.Normal(0., 1.))
    with pyro.plate("mu", 85):
        a =  pyro.sample("a", dist.Normal(mu_a, sigma_a))
        b =  pyro.sample("b", dist.Normal((0.1 * mu_b), sigma_b))
    with pyro.plate("data", N):
        y_hat = a[...,county] + x * b[...,county] + beta[...,0].unsqueeze(-1) * u + beta[...,1].unsqueeze(-1) * inter
        y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)

