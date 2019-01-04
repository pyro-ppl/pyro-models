# model file: ../example-models/ARM/Ch.3/kidiq_multi_preds.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'kid_score' in data, 'variable not found in data: key=kid_score'
    assert 'mom_iq' in data, 'variable not found in data: key=mom_iq'
    assert 'mom_hs' in data, 'variable not found in data: key=mom_hs'
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_iq = data["mom_iq"]
    mom_hs = data["mom_hs"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_iq = data["mom_iq"]
    mom_hs = data["mom_hs"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_iq = data["mom_iq"]
    mom_hs = data["mom_hs"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    sigma =  _pyro_sample(sigma, "sigma", "cauchy", [0, 2.5])
    kid_score =  _pyro_sample(kid_score, "kid_score", "normal", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,mom_hs])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,mom_iq])]), sigma], obs=kid_score)

