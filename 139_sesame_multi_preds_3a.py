# model file: ../example-models/ARM/Ch.10/sesame_multi_preds_3a.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'encouraged' in data, 'variable not found in data: key=encouraged'
    assert 'setting' in data, 'variable not found in data: key=setting'
    assert 'site' in data, 'variable not found in data: key=site'
    assert 'pretest' in data, 'variable not found in data: key=pretest'
    assert 'watched' in data, 'variable not found in data: key=watched'
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    setting = data["setting"]
    site = data["site"]
    pretest = data["pretest"]
    watched = data["watched"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    setting = data["setting"]
    site = data["site"]
    pretest = data["pretest"]
    watched = data["watched"]
    site2 = init_vector("site2", dims=(N)) # vector
    site3 = init_vector("site3", dims=(N)) # vector
    site4 = init_vector("site4", dims=(N)) # vector
    site5 = init_vector("site5", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):

        site2[i - 1] = _pyro_assign(site2[i - 1], _call_func("logical_eq", [_index_select(site, i - 1) ,2]))
        site3[i - 1] = _pyro_assign(site3[i - 1], _call_func("logical_eq", [_index_select(site, i - 1) ,3]))
        site4[i - 1] = _pyro_assign(site4[i - 1], _call_func("logical_eq", [_index_select(site, i - 1) ,4]))
        site5[i - 1] = _pyro_assign(site5[i - 1], _call_func("logical_eq", [_index_select(site, i - 1) ,5]))
    data["site2"] = site2
    data["site3"] = site3
    data["site4"] = site4
    data["site5"] = site5

def init_params(data, params):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    setting = data["setting"]
    site = data["site"]
    pretest = data["pretest"]
    watched = data["watched"]
    # initialize transformed data
    site2 = data["site2"]
    site3 = data["site3"]
    site4 = data["site4"]
    site5 = data["site5"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(8)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    setting = data["setting"]
    site = data["site"]
    pretest = data["pretest"]
    watched = data["watched"]
    # initialize transformed data
    site2 = data["site2"]
    site3 = data["site3"]
    site4 = data["site4"]
    site5 = data["site5"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    watched =  _pyro_sample(watched, "watched", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,encouraged])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,pretest])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,site2])]),_call_func("multiply", [_index_select(beta, 5 - 1) ,site3])]),_call_func("multiply", [_index_select(beta, 6 - 1) ,site4])]),_call_func("multiply", [_index_select(beta, 7 - 1) ,site5])]),_call_func("multiply", [_index_select(beta, 8 - 1) ,setting])]), sigma], obs=watched)

