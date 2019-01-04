# model file: ../example-models/ARM/Ch.7/earnings2.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earnings' in data, 'variable not found in data: key=earnings'
    assert 'height' in data, 'variable not found in data: key=height'
    assert 'sex' in data, 'variable not found in data: key=sex'
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]
    log_earnings = init_vector("log_earnings", dims=(N)) # vector
    male = init_vector("male", dims=(N)) # vector
    log_earnings = _pyro_assign(log_earnings, _call_func("log", [earnings]))
    male = _pyro_assign(male, _call_func("subtract", [2,sex]))
    data["log_earnings"] = log_earnings
    data["male"] = male

def init_params(data, params):
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]
    # initialize transformed data
    log_earnings = data["log_earnings"]
    male = data["male"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]
    # initialize transformed data
    log_earnings = data["log_earnings"]
    male = data["male"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log_earnings =  _pyro_sample(log_earnings, "log_earnings", "normal", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,height])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,male])]), sigma], obs=log_earnings)

