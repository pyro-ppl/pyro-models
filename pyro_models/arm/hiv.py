# model file: example-models/ARM/Ch.20/hiv.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'person' in data, 'variable not found in data: key=person'
    assert 'time' in data, 'variable not found in data: key=time'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    y = data["y"]
    # assign init values for parameters
    params["sigma_a1"] = pyro.sample("sigma_a1", dist.Uniform(0., 100.))
    params["sigma_a2"] = pyro.sample("sigma_a2", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"].long() - 1
    time = data["time"]
    y = data["y"]

    # init parameters
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    mu_a1 =  pyro.sample("mu_a1", dist.Normal(0., 1.))
    mu_a2 =  pyro.sample("mu_a2", dist.Normal(0., 1.))
    with pyro.plate('person', J):
        a1 =  pyro.sample("a1", dist.Normal(mu_a1, sigma_a1))
        a2 =  pyro.sample("a2", dist.Normal((0.1 * mu_a2), sigma_a2))
    with pyro.plate('data', N, dim=-1):
        y_hat = a1[...,person] + a2[...,person] * time
        y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)

