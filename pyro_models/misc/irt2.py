# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# model file: example-models/misc/irt/irt2.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'jj' in data, 'variable not found in data: key=jj'
    assert 'kk' in data, 'variable not found in data: key=kk'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"]
    kk = data["kk"]
    y = data["y"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"].long() - 1
    kk = data["kk"].long() - 1
    y = data["y"]

    with pyro.plate('alpha_', J):
        alpha =  pyro.sample("alpha", dist.Normal(0., 1.))
    with pyro.plate('beta_', K):
        beta =  pyro.sample("beta", dist.Normal(0., 1.))
        log_gamma =  pyro.sample("log_gamma", dist.Normal(0., 1.))
    delta =  pyro.sample("delta", dist.Normal(0.75, 1.))
    with pyro.plate('data', N):
        y = pyro.sample('y', dist.Bernoulli(logits= log_gamma[kk].exp()
                                            * (alpha[jj] - beta[kk] + delta)), obs=y)
