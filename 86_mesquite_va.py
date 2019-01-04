# model file: ../example-models/ARM/Ch.4/mesquite_va.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'weight' in data, 'variable not found in data: key=weight'
    assert 'diam1' in data, 'variable not found in data: key=diam1'
    assert 'diam2' in data, 'variable not found in data: key=diam2'
    assert 'canopy_height' in data, 'variable not found in data: key=canopy_height'
    assert 'group' in data, 'variable not found in data: key=group'
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    group = data["group"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    group = data["group"]
    log_weight = init_vector("log_weight", dims=(N)) # vector
    log_canopy_volume = init_vector("log_canopy_volume", dims=(N)) # vector
    log_canopy_area = init_vector("log_canopy_area", dims=(N)) # vector
    log_weight = _pyro_assign(log_weight, _call_func("log", [weight]))
    log_canopy_volume = _pyro_assign(log_canopy_volume, _call_func("log", [_call_func("elt_multiply", [_call_func("elt_multiply", [diam1,diam2]),canopy_height])]))
    log_canopy_area = _pyro_assign(log_canopy_area, _call_func("log", [_call_func("elt_multiply", [diam1,diam2])]))
    data["log_weight"] = log_weight
    data["log_canopy_volume"] = log_canopy_volume
    data["log_canopy_area"] = log_canopy_area

def init_params(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    group = data["group"]
    # initialize transformed data
    log_weight = data["log_weight"]
    log_canopy_volume = data["log_canopy_volume"]
    log_canopy_area = data["log_canopy_area"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    group = data["group"]
    # initialize transformed data
    log_weight = data["log_weight"]
    log_canopy_volume = data["log_canopy_volume"]
    log_canopy_area = data["log_canopy_area"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log_weight =  _pyro_sample(log_weight, "log_weight", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,log_canopy_volume])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,log_canopy_area])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,group])]), sigma], obs=log_weight)

