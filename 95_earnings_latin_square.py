# model file: ../example-models/ARM/Ch.13/earnings_latin_square.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_age' in data, 'variable not found in data: key=n_age'
    assert 'n_eth' in data, 'variable not found in data: key=n_eth'
    assert 'age' in data, 'variable not found in data: key=age'
    assert 'eth' in data, 'variable not found in data: key=eth'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"]
    eth = data["eth"]
    x = data["x"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"]
    eth = data["eth"]
    x = data["x"]
    y = data["y"]
    # assign init values for parameters
    params["sigma_a1"] = pyro.sample("sigma_a1", dist.Uniform(0., 100.))
    params["sigma_a2"] = pyro.sample("sigma_a2", dist.Uniform(0., 100.))
    params["sigma_b1"] = pyro.sample("sigma_b1", dist.Uniform(0., 100.))
    params["sigma_b2"] = pyro.sample("sigma_b2", dist.Uniform(0., 100.))
    params["sigma_c"] = pyro.sample("sigma_c", dist.Uniform(0., 100.))
    params["sigma_d"] = pyro.sample("sigma_d", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"].long() - 1
    eth = data["eth"].long() - 1
    x = data["x"]
    y = data["y"]

    # init parameters
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_b1 = params["sigma_b1"]
    sigma_b2 = params["sigma_b2"]
    sigma_c = params["sigma_c"]
    sigma_d = params["sigma_d"]
    sigma_y = params["sigma_y"]

    mu_a1 = pyro.sample('mu_a1', dist.Normal(0., 1.))
    mu_a2 = pyro.sample('mu_a2', dist.Normal(0., 1.))
    a1 = pyro.sample('a1', dist.Normal(10 * mu_a1, sigma_a1).expand([n_eth]))
    a2 = pyro.sample('a2', dist.Normal(10 * mu_a2, sigma_a2).expand([n_eth]))

    mu_b1 = pyro.sample('mu_b1', dist.Normal(0., 1.))
    mu_b2 = pyro.sample('mu_b2', dist.Normal(0., 1.))
    b1 = pyro.sample('b1', dist.Normal(10 * mu_b1, sigma_b1).expand([n_age]))
    b2 = pyro.sample('b2', dist.Normal(0.1* mu_b2, sigma_b2).expand([n_age]))

    mu_c = pyro.sample('mu_c', dist.Normal(0., 1.))
    c = pyro.sample('c', dist.Normal(10. * mu_c, sigma_c).expand([n_eth, n_age]))

    mu_d = pyro.sample('mu_d', dist.Normal(0., 1.))
    d = pyro.sample('d', dist.Normal(0.1 * mu_d, sigma_d).expand([n_eth, n_age]))

    y_hat = a1[eth] + a2[eth] * x + b1[age] + b2[age] * x +\
            c[eth, age] + d[eth, age] * x
    pyro.sample('y', dist.Normal(y_hat, sigma_y), obs=y)
