# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# model file: example-models/ARM/Ch.5/separation.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'x' in data, 'variable not found in data: key=x'
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]

    # init parameters
    beta = params["beta"]
    # model block
    with pyro.plate("data", N):
        y = pyro.sample('y', dist.Bernoulli(logits=beta[0] + beta[1] * x), obs=y)

