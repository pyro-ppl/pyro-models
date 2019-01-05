# model file: ../example-models/ARM/Ch.4/logearn_height.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    log_earn = init_vector("log_earn", dims=(N)) # vector
    log_earn = _pyro_assign(log_earn, _call_func("log", [earn]))
    data["log_earn"] = log_earn

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    # initialize transformed data
    log_earn = data["log_earn"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    # initialize transformed data
    log_earn = data["log_earn"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log_earn =  _pyro_sample(log_earn, "log_earn", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,height])]), sigma], obs=log_earn)

