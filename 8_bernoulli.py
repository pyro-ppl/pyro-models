# model file: ../example-models/basic_estimators/bernoulli.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["theta"] = pyro.sample("theta", dist.Uniform(0., 1))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    
    # init parameters
    theta = params["theta"]
    # initialize transformed parameters
    # model block

    theta =  _pyro_sample(theta, "theta", "beta", [1, 1])
    for n in range(1, to_int(N) + 1):
        y[n - 1] =  _pyro_sample(_index_select(y, n - 1) , "y[%d]" % (to_int(n-1)), "bernoulli", [theta], obs=_index_select(y, n - 1) )

