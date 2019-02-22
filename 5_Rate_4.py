# model file: example-models/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial/Rate_4.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'k' in data, 'variable not found in data: key=k'
    # initialize data
    n = data["n"]
    k = data["k"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    data = {k: torch.tensor(v).float() for k, v in data.items()}
    n = data["n"]
    k = data["k"]
    # model block
    theta =  pyro.sample("theta", dist.Beta(1., 1.))
    thetaprior =  pyro.sample("thetaprior", dist.Beta(1., 1.))
    k =  pyro.sample("k", dist.Binomial(n, theta), obs=k)

