# model file: ../example-models/ARM/Ch.5/wells_d100ars.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'arsenic' in data, 'variable not found in data: key=arsenic'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    dist100 = init_vector("dist100", dims=(N)) # vector
    dist100 = _pyro_assign(dist100, _call_func("divide", [dist,100.0]))
    data["dist100"] = dist100

def init_params(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    # initialize transformed data
    dist100 = data["dist100"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    # initialize transformed data
    dist100 = data["dist100"]
    # INIT parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    switched =  _pyro_sample(switched, "switched", "bernoulli_logit", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,dist100])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,arsenic])])], obs=switched)

