# model file: ../example-models/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial/Rate_2.stan
import torch
import pyro


def validate_data_def(data):
    assert 'n1' in data, 'variable not found in data: key=n1'
    assert 'n2' in data, 'variable not found in data: key=n2'
    assert 'k1' in data, 'variable not found in data: key=k1'
    assert 'k2' in data, 'variable not found in data: key=k2'
    # initialize data
    n1 = data["n1"]
    n2 = data["n2"]
    k1 = data["k1"]
    k2 = data["k2"]

def init_params(data, params):
    # initialize data
    n1 = data["n1"]
    n2 = data["n2"]
    k1 = data["k1"]
    k2 = data["k2"]
    # assign init values for parameters
    params["theta1"] = init_real("theta1", low=0, high=1) # real/double
    params["theta2"] = init_real("theta2", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    n1 = data["n1"]
    n2 = data["n2"]
    k1 = data["k1"]
    k2 = data["k2"]
    # INIT parameters
    theta1 = params["theta1"]
    theta2 = params["theta2"]
    # initialize transformed parameters
    delta = init_real("delta", low=-(1), high=1) # real/double
    delta = _pyro_assign(delta, (theta1 - theta2))
    # model block

    theta1 =  _pyro_sample(theta1, "theta1", "beta", [1, 1])
    theta2 =  _pyro_sample(theta2, "theta2", "beta", [1, 1])
    k1 =  _pyro_sample(k1, "k1", "binomial", [n1, theta1], obs=k1)
    k2 =  _pyro_sample(k2, "k2", "binomial", [n2, theta2], obs=k2)

