# model file: example-models/ARM/Ch.8/unemployment.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'y_lag' in data, 'variable not found in data: key=y_lag'
    # initialize data
    N = data["N"]
    y = data["y"]
    y_lag = data["y_lag"]

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0., 1000.))
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    y_lag = data["y_lag"]

    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    with pyro.plate("data", N):
        pyro.sample('obs', dist.Normal(beta[0] + beta[1] * y_lag, sigma), obs=y)

