# model file: ../example-models/ARM/Ch.8/unemployment.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'y_lag' in data, 'variable not found in data: key=y_lag'
    # initialize data
    N = data["N"]
    y = data["y"]
    y_lag = data["y_lag"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    y_lag = data["y_lag"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    y_lag = data["y_lag"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    y =  _pyro_sample(y, "y", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,y_lag])]), sigma], obs=y)

