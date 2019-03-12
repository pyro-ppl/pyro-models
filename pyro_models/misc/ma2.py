# model file: example-models/misc/moving-avg/ma2.stan
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
    return params

def model(data, params):
    # initialize data
    T = data["T"]
    y = data["y"]

    # model block
    mu =  pyro.sample("mu",  dist.HalfCauchy(2.5))
    theta =  pyro.sample("theta",  dist.HalfCauchy(2.5).expand([2]))
    sigma =  pyro.sample("sigma",  dist.HalfCauchy(2.5))
    with torch.no_grad():
        epsilon = init_vector("epsilon", dims=T)
        epsilon[0] = y[0] - mu
        epsilon[1] = y[1] - mu - theta[0] * epsilon[0]
        for t in range(2, T):
            epsilon[t] = y[t] - mu - theta[0] * epsilon[t - 1] - theta[1] * epsilon[t - 2]
    for t in range(2, T):
        pyro.sample("y_{}".format(t), dist.Normal(mu + theta[0] * epsilon[t - 1] + \
            theta[1] * epsilon[t - 2], sigma), obs=y[t])
