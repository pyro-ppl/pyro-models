# model file: ../example-models/ARM/Ch.10/sesame_one_pred_a.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'encouraged' in data, 'variable not found in data: key=encouraged'
    assert 'watched' in data, 'variable not found in data: key=watched'
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    watched =  _pyro_sample(watched, "watched", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,encouraged])]), sigma], obs=watched)

