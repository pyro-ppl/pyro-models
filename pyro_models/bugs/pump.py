# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# model file: example-models/bugs_examples/vol1/pump/pump.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 't' in data, 'variable not found in data: key=t'
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    x = data["x"]
    t = data["t"]

    alpha =  pyro.sample("alpha", dist.Exponential(1.0))
    beta =  pyro.sample("beta", dist.Gamma(0.1, 1.0))
    with pyro.plate('data', N):
        theta =  pyro.sample("theta", dist.Gamma(alpha, beta))
        x =  pyro.sample("x", dist.Poisson(theta * t), obs=x)

