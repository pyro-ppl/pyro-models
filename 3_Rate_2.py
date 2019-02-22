# model file: example-models/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial/Rate_2.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

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

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    data = {k: torch.tensor(v).float() for k, v in data.items()}
    n1 = data["n1"]
    n2 = data["n2"]
    k1 = data["k1"]
    k2 = data["k2"]
    # init parameters
    theta1 =  pyro.sample("theta1", dist.Beta(1., 1.))
    theta2 =  pyro.sample("theta2", dist.Beta(1., 1.))
    pyro.sample("k1", dist.Binomial(n1, theta1), obs=k1)
    pyro.sample("k2", dist.Binomial(n2, theta2), obs=k2)
