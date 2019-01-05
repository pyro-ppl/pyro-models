# model file: ../example-models/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial/Rate_4.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'k' in data, 'variable not found in data: key=k'
    # initialize data
    n = data["n"]
    k = data["k"]

def init_params(data):
    params = {}
    # initialize data
    n = data["n"]
    k = data["k"]
    # assign init values for parameters
    params["theta"] = pyro.sample("theta", dist.Uniform(0., 1))
    params["thetaprior"] = pyro.sample("thetaprior", dist.Uniform(0., 1))

    return params

def model(data, params):
    # initialize data
    n = data["n"]
    k = data["k"]
    
    # init parameters
    theta = params["theta"]
    thetaprior = params["thetaprior"]
    # initialize transformed parameters
    # model block

    theta =  _pyro_sample(theta, "theta", "beta", [1, 1])
    thetaprior =  _pyro_sample(thetaprior, "thetaprior", "beta", [1, 1])
    k =  _pyro_sample(k, "k", "binomial", [n, theta], obs=k)

