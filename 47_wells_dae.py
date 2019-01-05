# model file: ../example-models/ARM/Ch.5/wells_dae.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'arsenic' in data, 'variable not found in data: key=arsenic'
    assert 'educ' in data, 'variable not found in data: key=educ'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    dist100 = init_vector("dist100", dims=(N)) # vector
    educ4 = init_vector("educ4", dims=(N)) # vector
    dist100 = _pyro_assign(dist100., _call_func("divide", [dist,100.0]))
    educ4 = _pyro_assign(educ4, _call_func("divide", [educ,4.0]))
    data["dist100"] = dist100
    data["educ4"] = educ4

def init_params(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    # initialize transformed data
    dist100 = data["dist100"]
    educ4 = data["educ4"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    # initialize transformed data
    dist100 = data["dist100"]
    educ4 = data["educ4"]
    
    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    switched =  _pyro_sample(switched, "switched", "bernoulli_logit", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,dist100])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,arsenic])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,educ4])])], obs=switched)

