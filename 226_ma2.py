# model file: ../example-models/misc/moving-avg/ma2.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



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
    params["mu"] = pyro.sample("mu"))
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))
    params["theta"] = init_vector("theta", dims=(2)) # vector

    return params

def model(data, params):
    # initialize data
    T = data["T"]
    y = data["y"]
    
    # init parameters
    mu = params["mu"]
    sigma = params["sigma"]
    theta = params["theta"]
    # initialize transformed parameters
    epsilon = init_vector("epsilon", dims=(T)) # vector
    epsilon[1 - 1] = _pyro_assign(epsilon[1 - 1], (_index_select(y, 1 - 1)  - mu))
    epsilon[2 - 1] = _pyro_assign(epsilon[2 - 1], ((_index_select(y, 2 - 1)  - mu) - (_index_select(theta, 1 - 1)  * _index_select(epsilon, 1 - 1) )))
    for t in range(3, to_int(T) + 1):
        epsilon[t - 1] = _pyro_assign(epsilon[t - 1], (((_index_select(y, t - 1)  - mu) - (_index_select(theta, 1 - 1)  * _index_select(epsilon, (t - 1) - 1) )) - (_index_select(theta, 2 - 1)  * _index_select(epsilon, (t - 2) - 1) )))
    # model block

    mu =  _pyro_sample(mu, "mu", "cauchy", [0., 2.5])
    theta =  _pyro_sample(theta, "theta", "cauchy", [0., 2.5])
    sigma =  _pyro_sample(sigma, "sigma", "cauchy", [0., 2.5])
    for t in range(3, to_int(T) + 1):
        y[t - 1] =  _pyro_sample(_index_select(y, t - 1) , "y[%d]" % (to_int(t-1)), "normal", [((mu + (_index_select(theta, 1 - 1)  * _index_select(epsilon, (t - 1) - 1) )) + (_index_select(theta, 2 - 1)  * _index_select(epsilon, (t - 2) - 1) )), sigma], obs=_index_select(y, t - 1) )

