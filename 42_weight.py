# model file: ../example-models/ARM/Ch.18/weight.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'weight' in data, 'variable not found in data: key=weight'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]
    c_height = height - height.mean(0)
    data["c_height"] = c_height

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]
    # assign init values for parameters
    params["a"] = pyro.sample("a", dist.Uniform(0., 100.))
    params["b"] = pyro.sample("b", dist.Uniform(0., 100.))
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]
    # initialize transformed data
    c_height = data["c_height"]

    # init parameters
    a = params["a"]
    b = params["b"]
    sigma = params["sigma"]
    # initialize transformed parameters

    # model block
    weight = pyro.sample('weight', dist.Normal(a + b * c_height, sigma), obs=weight)
