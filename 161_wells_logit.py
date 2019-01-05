# model file: ../example-models/ARM/Ch.6/wells_logit.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'switc' in data, 'variable not found in data: key=switc'
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    dist100 = init_vector("dist100", dims=(N)) # vector
    dist100 = _pyro_assign(dist100., _call_func("divide", [dist,100.0]))
    data["dist100"] = dist100

def init_params(data, params):
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    # initialize transformed data
    dist100 = data["dist100"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    dist = data["dist"]
    switc = data["switc"]
    # initialize transformed data
    dist100 = data["dist100"]
    
    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    switc =  _pyro_sample(switc, "switc", "bernoulli_logit", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,dist100])])], obs=switc)

