# model file: ../example-models/ARM/Ch.7/earnings_interactions.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
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
    check_constraints(N, low=0, dims=[1])
    check_constraints(earnings, dims=[N])
    check_constraints(height, dims=[N])
    check_constraints(sex, dims=[N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]
    log_earnings = init_vector("log_earnings", dims=(N)) # vector
    male = init_vector("male", dims=(N)) # vector
    height_male_inter = init_vector("height_male_inter", dims=(N)) # vector
    log_earnings = _pyro_assign(log_earnings, _call_func("log", [earnings]))
    male = _pyro_assign(male, _call_func("subtract", [2,sex]))
    height_male_inter = _pyro_assign(height_male_inter, _call_func("elt_multiply", [height,male]))
    data["log_earnings"] = log_earnings
    data["male"] = male
    data["height_male_inter"] = height_male_inter

def init_params(data, params):
    # initialize data
    N = data["N"]
    earnings = data["earnings"]
    height = data["height"]
    sex = data["sex"]
    # initialize transformed data
    log_earnings = data["log_earnings"]
    male = data["male"]
    height_male_inter = data["height_male_inter"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
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
    height_male_inter = data["height_male_inter"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log_earnings =  _pyro_sample(log_earnings, "log_earnings", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,height])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,male])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,height_male_inter])]), sigma], obs=log_earnings)

