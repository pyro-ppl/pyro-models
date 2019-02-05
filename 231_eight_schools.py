# model file: ../example-models/misc/eight_schools/eight_schools.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'sigma' in data, 'variable not found in data: key=sigma'
    # initialize data
    J = data["J"]
    y = data["y"]
    sigma = data["sigma"]

def init_params(data):
    params = {}
    # initialize data
    J = data["J"]
    y = data["y"]
    sigma = data["sigma"]
    # assign init values for parameters
    params["mu"] = pyro.sample("mu", dist.Normal(0., 1.))
    params["theta"] = init_vector("theta", dims=(J))
    return params

def model(data, params):
    # initialize data
    J = data["J"]
    y = data["y"]
    sigma = data["sigma"]

    # init parameters
    mu = params["mu"]
    theta = params["theta"]
    tau = pyro.sample('tau', dist.HalfCauchy(2.5))
    sigma = pyro.sample('sigma', dist.HalfCauchy(2.5))
    # initialize transformed parameters
    with pyro.plate('data', J):
        y = pyro.sample('y', dist.Normal(theta, sigma), obs=y)
