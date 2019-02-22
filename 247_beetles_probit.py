# model file: example-models/bugs_examples/vol2/beetles/beetles_probit.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'r' in data, 'variable not found in data: key=r'
    assert 'x' in data, 'variable not found in data: key=x'
    # initialize data
    N = data["N"]
    n = data["n"]
    r = data["r"]
    x = data["x"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    n = data["n"]
    r = data["r"]
    x = data["x"]
    mean_x = x.mean()
    centered_x = x - mean_x
    data["centered_x"] = centered_x
    data["mean_x"] = mean_x

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n = data["n"]
    r = data["r"]
    x = data["x"]
    # initialize transformed data
    centered_x = data["centered_x"]
    mean_x = data["mean_x"]

    # initialize transformed parameters
    alpha_star =  pyro.sample("alpha_star", dist.Normal(0., 1.0))
    beta =  pyro.sample("beta", dist.Normal(0., 10000.0))
    with pyro.plate('data', N):
        p = dist.Normal(0., 1.).cdf(alpha_star + beta * centered_x)
        r =  pyro.sample("r", dist.Binomial(n, p), obs=r)

