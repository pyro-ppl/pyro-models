# model file: ../example-models/ARM/Ch.23/electric_one_pred.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'post_test' in data, 'variable not found in data: key=post_test'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    # initialize data
    N = data["N"]
    post_test = data["post_test"]
    treatment = data["treatment"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    post_test = data["post_test"]
    treatment = data["treatment"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    post_test = data["post_test"]
    treatment = data["treatment"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    post_test =  _pyro_sample(post_test, "post_test", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,treatment])]), sigma], obs=post_test)

