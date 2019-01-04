# model file: ../example-models/ARM/Ch.10/sesame_one_pred_2b.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'watched_hat' in data, 'variable not found in data: key=watched_hat'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    watched_hat = data["watched_hat"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    watched_hat = data["watched_hat"]
    y = data["y"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    watched_hat = data["watched_hat"]
    y = data["y"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    y =  _pyro_sample(y, "y", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,watched_hat])]), sigma], obs=y)

