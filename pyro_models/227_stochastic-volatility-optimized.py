# model file: example-models/misc/moving-avg/stochastic-volatility-optimized.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    T = data["T"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    T = data["T"]
    y = data["y"]
    # assign init values for parameters
    params["phi"] = pyro.sample("phi", dist.Uniform(-(1), 1))

    return params

def model(data, params):
    # initialize data
    T = data["T"]
    y = data["y"]

    # init parameters
    phi = params["phi"]
    # initialize transformed parameters
    h = init_vector("h", dims=(T)) # vector
    # model block

    sigma =  pyro.sample("sigma", dist.HalfCauchy(5.))
    mu =  pyro.sample("mu", dist.Cauchy(0., 10.))
    h_std =  pyro.sample("h_std", dist.Normal(0., 1.).expand([T]))
    with torch.no_grad():
        h = h_std * sigma
        h[0] = h[0] / torch.sqrt(1. - phi * phi)
        h = h + mu
        for t in range(1, T):
            h[t] = h[t] + phi * (h[t-1] - mu);
    y = pyro.sample(y, dist.Normal(0., (h / 2.).exp()), obs=y)

