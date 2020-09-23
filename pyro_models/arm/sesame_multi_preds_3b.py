# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# model file: example-models/ARM/Ch.10/sesame_multi_preds_3b.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'pretest' in data, 'variable not found in data: key=pretest'
    assert 'setting' in data, 'variable not found in data: key=setting'
    assert 'site' in data, 'variable not found in data: key=site'
    assert 'watched_hat' in data, 'variable not found in data: key=watched_hat'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    pretest = data["pretest"]
    setting = data["setting"]
    site = data["site"]
    watched_hat = data["watched_hat"]
    y = data["y"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    pretest = data["pretest"]
    setting = data["setting"]
    site = data["site"]
    watched_hat = data["watched_hat"]
    y = data["y"]
    site2 = site == 2
    site3 = site == 3
    site4 = site == 4
    site5 = site == 5
    data["site2"] = site2.float()
    data["site3"] = site3.float()
    data["site4"] = site4.float()
    data["site5"] = site5.float()

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(8)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    pretest = data["pretest"]
    setting = data["setting"]
    site = data["site"]
    watched_hat = data["watched_hat"]
    y = data["y"]
    # initialize transformed data
    site2 = data["site2"]
    site3 = data["site3"]
    site4 = data["site4"]
    site5 = data["site5"]

    # init parameters
    beta = params["beta"]
    sigma = pyro.sample('sigma', dist.HalfCauchy(2.5))
    # initialize transformed parameters
    # model block

    with pyro.plate('data', N):
        y = pyro.sample('obs', dist.Normal(beta[0] + beta[1] * watched_hat + \
                        beta[2] * pretest + beta[3] * site2 + beta[4] * site3 + \
                        beta[5] * site4 + beta[6] * site5 + beta[7] * setting, sigma), obs=y)
