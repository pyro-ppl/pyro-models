# model file: example-models/ARM/Ch.4/logearn_height_male.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    log_earn = torch.log(earn)
    data["log_earn"] = log_earn

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(3)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    height = data["height"]
    male = data['male']
    # initialize transformed data
    log_earn = data["log_earn"]

    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    sigma =  pyro.sample("sigma", dist.HalfCauchy(torch.tensor(2.5)))
    with pyro.plate("data", N):    
        log_earn = pyro.sample('obs', dist.Normal(beta[0] + beta[1] * height + beta[2] * male, sigma), obs=log_earn)

