# model file: ../example-models/ARM/Ch.4/mesquite_log.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'weight' in data, 'variable not found in data: key=weight'
    assert 'diam1' in data, 'variable not found in data: key=diam1'
    assert 'diam2' in data, 'variable not found in data: key=diam2'
    assert 'canopy_height' in data, 'variable not found in data: key=canopy_height'
    assert 'total_height' in data, 'variable not found in data: key=total_height'
    assert 'density' in data, 'variable not found in data: key=density'
    assert 'group' in data, 'variable not found in data: key=group'
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    total_height = data["total_height"]
    density = data["density"]
    group = data["group"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    total_height = data["total_height"]
    density = data["density"]
    group = data["group"]
    log_weight = init_vector("log_weight", dims=(N)) # vector
    log_diam1 = init_vector("log_diam1", dims=(N)) # vector
    log_diam2 = init_vector("log_diam2", dims=(N)) # vector
    log_canopy_height = init_vector("log_canopy_height", dims=(N)) # vector
    log_total_height = init_vector("log_total_height", dims=(N)) # vector
    log_density = init_vector("log_density", dims=(N)) # vector
    log_weight = _pyro_assign(log_weight, _call_func("log", [weight]))
    log_diam1 = _pyro_assign(log_diam1, _call_func("log", [diam1]))
    log_diam2 = _pyro_assign(log_diam2, _call_func("log", [diam2]))
    log_canopy_height = _pyro_assign(log_canopy_height, _call_func("log", [canopy_height]))
    log_total_height = _pyro_assign(log_total_height, _call_func("log", [total_height]))
    log_density = _pyro_assign(log_density, _call_func("log", [density]))
    data["log_weight"] = log_weight
    data["log_diam1"] = log_diam1
    data["log_diam2"] = log_diam2
    data["log_canopy_height"] = log_canopy_height
    data["log_total_height"] = log_total_height
    data["log_density"] = log_density

def init_params(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    total_height = data["total_height"]
    density = data["density"]
    group = data["group"]
    # initialize transformed data
    log_weight = data["log_weight"]
    log_diam1 = data["log_diam1"]
    log_diam2 = data["log_diam2"]
    log_canopy_height = data["log_canopy_height"]
    log_total_height = data["log_total_height"]
    log_density = data["log_density"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(7)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    total_height = data["total_height"]
    density = data["density"]
    group = data["group"]
    # initialize transformed data
    log_weight = data["log_weight"]
    log_diam1 = data["log_diam1"]
    log_diam2 = data["log_diam2"]
    log_canopy_height = data["log_canopy_height"]
    log_total_height = data["log_total_height"]
    log_density = data["log_density"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    log_weight =  _pyro_sample(log_weight, "log_weight", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,log_diam1])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,log_diam2])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,log_canopy_height])]),_call_func("multiply", [_index_select(beta, 5 - 1) ,log_total_height])]),_call_func("multiply", [_index_select(beta, 6 - 1) ,log_density])]),_call_func("multiply", [_index_select(beta, 7 - 1) ,group])]), sigma], obs=log_weight)

