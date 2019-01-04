# model file: ../example-models/ARM/Ch.7/congress.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'incumbency_88' in data, 'variable not found in data: key=incumbency_88'
    assert 'vote_86' in data, 'variable not found in data: key=vote_86'
    assert 'vote_88' in data, 'variable not found in data: key=vote_88'
    # initialize data
    N = data["N"]
    incumbency_88 = data["incumbency_88"]
    vote_86 = data["vote_86"]
    vote_88 = data["vote_88"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    incumbency_88 = data["incumbency_88"]
    vote_86 = data["vote_86"]
    vote_88 = data["vote_88"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    incumbency_88 = data["incumbency_88"]
    vote_86 = data["vote_86"]
    vote_88 = data["vote_88"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    vote_88 =  _pyro_sample(vote_88, "vote_88", "normal", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,vote_86])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,incumbency_88])]), sigma], obs=vote_88)

