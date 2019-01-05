# model file: ../example-models/ARM/Ch.7/earnings1.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn_pos' in data, 'variable not found in data: key=earn_pos'
    assert 'height' in data, 'variable not found in data: key=height'
    assert 'male' in data, 'variable not found in data: key=male'
    # initialize data
    N = data["N"]
    earn_pos = data["earn_pos"]
    height = data["height"]
    male = data["male"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    earn_pos = data["earn_pos"]
    height = data["height"]
    male = data["male"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    earn_pos = data["earn_pos"]
    height = data["height"]
    male = data["male"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    earn_pos =  _pyro_sample(earn_pos, "earn_pos", "bernoulli_logit", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,height])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,male])])], obs=earn_pos)

