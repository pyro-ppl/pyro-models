# model file: example-models/ARM/Ch.4/logearn_interaction.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'height' in data, 'variable not found in data: key=height'
    assert 'male' in data, 'variable not found in data: key=male'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]
    log_earn = torch.log(earn);
    inter    = height * male;
    data["log_earn"] = log_earn
    data["inter"] = inter

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    height = data["height"]
    male = data["male"]
    # initialize transformed data
    log_earn = data["log_earn"]
    inter = data["inter"]

    # init parameters
    beta = params["beta"]
    sigma =  pyro.sample("sigma", dist.HalfCauchy(torch.tensor(2.5)))
    with pyro.plate("data", N):
        log_earn = pyro.sample('obs', dist.Normal(beta[0] + beta[1] * height + beta[2] * male + beta[3] * inter, sigma), obs=log_earn)
