# model file: ../example-models/ARM/Ch.4/log10earn_height.stan
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
    log10_earn = init_vector("log10_earn", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):

        log10_earn[i - 1] = _pyro_assign(log10_earn[i - 1], _call_func("log10", [_index_select(earn, i - 1) ]))
    data["log10_earn"] = log10_earn

def init_params(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    # initialize transformed data
    log10_earn = data["log10_earn"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    # initialize transformed data
    log10_earn = data["log10_earn"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log10_earn =  _pyro_sample(log10_earn, "log10_earn", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,height])]), sigma], obs=log10_earn)

