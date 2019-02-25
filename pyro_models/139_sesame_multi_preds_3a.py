# model file: example-models/ARM/Ch.10/sesame_multi_preds_3a.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



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
    site2 = site == 2
    site3 = site == 3
    site4 = site == 4
    site5 = site == 5
    data["site2"] = site2.float()
    data["site3"] = site3.float()
    data["site4"] = site4.float()
    data["site5"] = site5.float()

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(8)) # vector
    return params

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
    sigma = pyro.sample('sigma', dist.HalfCauchy(2.5))
    # initialize transformed parameters
    # model block
    with pyro.plate('data', N):
        watched = pyro.sample('obs', dist.Normal(beta[0] + beta[1] * encouraged + \
                              beta[2] * pretest + beta[3] * site2 + beta[4] * site3 + \
                              beta[5] * site4 + beta[6] * site5 + beta[7] * setting, sigma), obs=watched)

