# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# model file: example-models/bugs_examples/vol1/dyes/dyes.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'BATCHES' in data, 'variable not found in data: key=BATCHES'
    assert 'SAMPLES' in data, 'variable not found in data: key=SAMPLES'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    BATCHES = data["BATCHES"]
    SAMPLES = data["SAMPLES"]
    y = data["y"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    BATCHES = data["BATCHES"]
    SAMPLES = data["SAMPLES"]
    y = data["y"]
    # model block
    theta =  pyro.sample("theta", dist.Normal(0.0, 100000.0))
    tau_between =  pyro.sample("tau_between", dist.Gamma(0.001, 0.001))
    sigma_between = 1 / tau_between.sqrt()
    tau_within =  pyro.sample("tau_within", dist.Gamma(0.001, 0.001))
    sigma_within= 1 / tau_within.sqrt()
    with pyro.plate('batches', BATCHES, dim=-2):
        mu =  pyro.sample("mu", dist.Normal(theta, sigma_between))
        with pyro.plate('data', SAMPLES, dim=-1):
            y = pyro.sample('y', dist.Normal(mu, sigma_within), obs=y)
