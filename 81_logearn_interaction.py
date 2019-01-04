# model file: ../example-models/ARM/Ch.4/logearn_interaction.stan
import torch
import pyro


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
    inter = init_vector("inter", dims=(N)) # vector
    log_earn = _pyro_assign(log_earn, _call_func("log", [earn]))
    inter = _pyro_assign(inter, _call_func("elt_multiply", [height,male]))
    data["log_earn"] = log_earn
    data["inter"] = inter

def init_params(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]
    # initialize transformed data
    log_earn = data["log_earn"]
    inter = data["inter"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]
    # initialize transformed data
    log_earn = data["log_earn"]
    inter = data["inter"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log_earn =  _pyro_sample(log_earn, "log_earn", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,height])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,male])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,inter])]), sigma], obs=log_earn)

