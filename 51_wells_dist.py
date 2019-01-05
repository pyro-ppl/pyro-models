# model file: ../example-models/ARM/Ch.5/wells_dist.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    
    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    switched =  _pyro_sample(switched, "switched", "bernoulli_logit", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,dist])])], obs=switched)

