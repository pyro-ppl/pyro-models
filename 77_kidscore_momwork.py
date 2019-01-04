# model file: ../example-models/ARM/Ch.4/kidscore_momwork.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'kid_score' in data, 'variable not found in data: key=kid_score'
    assert 'mom_work' in data, 'variable not found in data: key=mom_work'
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_work = data["mom_work"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_work = data["mom_work"]
    work2 = init_vector("work2", dims=(N)) # vector
    work3 = init_vector("work3", dims=(N)) # vector
    work4 = init_vector("work4", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):

        work2[i - 1] = _pyro_assign(work2[i - 1], _call_func("logical_eq", [_index_select(mom_work, i - 1) ,2]))
        work3[i - 1] = _pyro_assign(work3[i - 1], _call_func("logical_eq", [_index_select(mom_work, i - 1) ,3]))
        work4[i - 1] = _pyro_assign(work4[i - 1], _call_func("logical_eq", [_index_select(mom_work, i - 1) ,4]))
    data["work2"] = work2
    data["work3"] = work3
    data["work4"] = work4

def init_params(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_work = data["mom_work"]
    # initialize transformed data
    work2 = data["work2"]
    work3 = data["work3"]
    work4 = data["work4"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_work = data["mom_work"]
    # initialize transformed data
    work2 = data["work2"]
    work3 = data["work3"]
    work4 = data["work4"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    kid_score =  _pyro_sample(kid_score, "kid_score", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,work2])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,work3])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,work4])]), sigma], obs=kid_score)

