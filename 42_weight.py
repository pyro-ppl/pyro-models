# model file: ../example-models/ARM/Ch.18/weight.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'weight' in data, 'variable not found in data: key=weight'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]
    c_height = init_vector("c_height", dims=(N)) # vector
    c_height = _pyro_assign(c_height, _call_func("subtract", [height,_call_func("mean", [height])]))
    data["c_height"] = c_height

def init_params(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]
    # initialize transformed data
    c_height = data["c_height"]
    # assign init values for parameters
    params["a"] = init_real("a") # real/double
    params["b"] = init_real("b") # real/double
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    height = data["height"]
    # initialize transformed data
    c_height = data["c_height"]
    # INIT parameters
    a = params["a"]
    b = params["b"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    weight =  _pyro_sample(weight, "weight", "normal", [_call_func("add", [a,_call_func("multiply", [b,c_height])]), sigma], obs=weight)

