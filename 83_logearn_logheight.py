# model file: ../example-models/ARM/Ch.4/logearn_logheight.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'height' in data, 'variable not found in data: key=height'
    assert 'male' in data, 'variable not found in data: key=male'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]
    log_earn = init_vector("log_earn", dims=(N)) # vector
    log_height = init_vector("log_height", dims=(N)) # vector
    log_earn = _pyro_assign(log_earn, _call_func("log", [earn]))
    log_height = _pyro_assign(log_height, _call_func("log", [height]))
    data["log_earn"] = log_earn
    data["log_height"] = log_height

def init_params(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]
    # initialize transformed data
    log_earn = data["log_earn"]
    log_height = data["log_height"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]
    # initialize transformed data
    log_earn = data["log_earn"]
    log_height = data["log_height"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log_earn =  _pyro_sample(log_earn, "log_earn", "normal", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,log_height])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,male])]), sigma], obs=log_earn)

