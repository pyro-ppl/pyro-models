# model file: ../example-models/ARM/Ch.5/separation.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'x' in data, 'variable not found in data: key=x'
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    
    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    y =  _pyro_sample(y, "y", "bernoulli_logit", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,x])])], obs=y)

