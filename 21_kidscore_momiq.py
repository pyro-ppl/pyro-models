# model file: ../example-models/ARM/Ch.3/kidscore_momiq.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'kid_score' in data, 'variable not found in data: key=kid_score'
    assert 'mom_iq' in data, 'variable not found in data: key=mom_iq'
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_iq = data["mom_iq"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_iq = data["mom_iq"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_iq = data["mom_iq"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    sigma =  _pyro_sample(sigma, "sigma", "cauchy", [0., 2.5])
    kid_score =  _pyro_sample(kid_score, "kid_score", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,mom_iq])]), sigma], obs=kid_score)

