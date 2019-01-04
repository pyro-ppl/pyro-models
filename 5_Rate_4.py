# model file: ../example-models/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial/Rate_4.stan
import torch
import pyro


def validate_data_def(data):
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'k' in data, 'variable not found in data: key=k'
    # initialize data
    n = data["n"]
    k = data["k"]

def init_params(data, params):
    # initialize data
    n = data["n"]
    k = data["k"]
    # assign init values for parameters
    params["theta"] = init_real("theta", low=0, high=1) # real/double
    params["thetaprior"] = init_real("thetaprior", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    n = data["n"]
    k = data["k"]
    # INIT parameters
    theta = params["theta"]
    thetaprior = params["thetaprior"]
    # initialize transformed parameters
    # model block

    theta =  _pyro_sample(theta, "theta", "beta", [1, 1])
    thetaprior =  _pyro_sample(thetaprior, "thetaprior", "beta", [1, 1])
    k =  _pyro_sample(k, "k", "binomial", [n, theta], obs=k)

