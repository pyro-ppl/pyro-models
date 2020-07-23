# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# model file: example-models/misc/irt/irt_multilevel.stan
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
    params["sigma_alpha"] = pyro.sample("sigma_alpha", dist.HalfCauchy(2.5))
    params["sigma_beta"] = pyro.sample("sigma_beta", dist.HalfCauchy(2.5))

    return params

def model(data, params):
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"].long() - 1
    kk = data["kk"].long() - 1
    y = data["y"]

    # init parameters
    sigma_alpha = params["sigma_alpha"]
    sigma_beta = params["sigma_beta"]
    # initialize transformed parameters
    # model block

    with pyro.plate('alpha_', J):
        alpha =  pyro.sample("alpha", dist.Normal(0., sigma_alpha))
    with pyro.plate('beta_', K):
        beta =  pyro.sample("beta", dist.Normal(0., sigma_beta))
    delta =  pyro.sample("delta", dist.Normal(0.75, 1.))
    with pyro.plate('data', N):
        y = pyro.sample('y', dist.Bernoulli(logits=alpha[jj] - beta[kk] + delta), obs=y)
