# model file: ../example-models/ARM/Ch.10/ideo_reparam.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'party' in data, 'variable not found in data: key=party'
    assert 'score1' in data, 'variable not found in data: key=score1'
    assert 'z1' in data, 'variable not found in data: key=z1'
    assert 'z2' in data, 'variable not found in data: key=z2'
    # initialize data
    N = data["N"]
    party = data["party"]
    score1 = data["score1"]
    z1 = data["z1"]
    z2 = data["z2"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    party = data["party"]
    score1 = data["score1"]
    z1 = data["z1"]
    z2 = data["z2"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    party = data["party"]
    score1 = data["score1"]
    z1 = data["z1"]
    z2 = data["z2"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    score1 =  _pyro_sample(score1, "score1", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,party])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,z1])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,z2])]), sigma], obs=score1)

