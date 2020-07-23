# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# model file: example-models/ARM/Ch.8/roaches_overdispersion.stan
import torch
import pyro
import pyro.distributions as dist
import sys

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'roach1' in data, 'variable not found in data: key=roach1'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'senior' in data, 'variable not found in data: key=senior'
    assert 'exposure2' in data, 'variable not found in data: key=exposure2'

def transformed_data(data):
    # initialize data
    exposure2 = data["exposure2"]
    log_expo = torch.log(exposure2)
    data["log_expo"] = log_expo

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    roach1 = data["roach1"]
    treatment = data["treatment"]
    senior = data["senior"]
    log_expo = data["log_expo"]

    # init parameters
    beta = params["beta"]

    # model block
    # Tau is the global precision
    # tau = pyro.sample('tau', dist.Gamma(concentration=0.001, rate=0.001))
    # sigma = tau.sqrt().reciprocal()

    # NOTE: Had to made change from Stan model here!
    sigma = pyro.sample('sigma', dist.HalfCauchy(2.5))

    with pyro.plate("data", N):
        # NOTE: the zeros_like is here since tau might be in plate
        lambdah = pyro.sample('lambda', dist.Normal(loc=torch.zeros_like(sigma), scale=sigma))
        log_rate = lambdah + log_expo + beta[..., 0] + beta[..., 1] * roach1 + beta[..., 2] * senior + beta[..., 3] * treatment

        # NOTE: Do we have to worry about under-/overflow here due to exp?
        pyro.sample('y', dist.Poisson(rate=log_rate.exp()+1e-8), obs=y)
