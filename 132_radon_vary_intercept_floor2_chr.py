# model file: ../example-models/ARM/Ch.21/radon_vary_intercept_floor2_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'county' in data, 'variable not found in data: key=county'
    assert 'u' in data, 'variable not found in data: key=u'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'x_mean' in data, 'variable not found in data: key=x_mean'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    N = data["N"]
    county = data["county"]
    u = data["u"]
    x = data["x"]
    x_mean = data["x_mean"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    J = data["J"]
    N = data["N"]
    county = data["county"]
    u = data["u"]
    x = data["x"]
    x_mean = data["x_mean"]
    y = data["y"]
    # assign init values for parameters
    params["b"] = init_vector("b", dims=(3)) # vector
    params["eta"] = init_vector("eta", dims=(J)) # vector
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    J = data["J"]
    N = data["N"]
    county = data["county"].long() - 1
    u = data["u"]
    x = data["x"]
    x_mean = data["x_mean"]
    y = data["y"]

    # init parameters
    b = params["b"]
    eta = params["eta"]
    sigma_a = params["sigma_a"]
    sigma_y = params["sigma_y"]

    # model block
    mu_a =  pyro.sample("mu_a", dist.Normal(0., 1))
    a = mu_a + sigma_a * eta
    y_hat = a[county] + u * b[0] + x * b[1] + x_mean * b[2]
    eta =  pyro.sample("eta", dist.Normal(0., 1))
    b =  pyro.sample("b", dist.Normal(0., 1))
    y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)

