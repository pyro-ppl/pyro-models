# model file: ../example-models/ARM/Ch.9/electric_trpre.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'post_test' in data, 'variable not found in data: key=post_test'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'pre_test' in data, 'variable not found in data: key=pre_test'
    # initialize data
    N = data["N"]
    post_test = data["post_test"]
    treatment = data["treatment"]
    pre_test = data["pre_test"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    post_test = data["post_test"]
    treatment = data["treatment"]
    pre_test = data["pre_test"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    post_test = data["post_test"]
    treatment = data["treatment"]
    pre_test = data["pre_test"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    post_test =  _pyro_sample(post_test, "post_test", "normal", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,treatment])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,pre_test])]), sigma], obs=post_test)

