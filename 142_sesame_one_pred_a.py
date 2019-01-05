# model file: ../example-models/ARM/Ch.10/sesame_one_pred_a.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'encouraged' in data, 'variable not found in data: key=encouraged'
    assert 'watched' in data, 'variable not found in data: key=watched'
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    encouraged = data["encouraged"]
    watched = data["watched"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    watched =  _pyro_sample(watched, "watched", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,encouraged])]), sigma], obs=watched)

