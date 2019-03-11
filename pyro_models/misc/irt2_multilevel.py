# model file: example-models/misc/irt/irt2_multilevel.stan
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
    sigma_alpha =  pyro.sample("sigma_alpha", dist.HalfCauchy(5.))
    sigma_beta =  pyro.sample("sigma_beta", dist.HalfCauchy(5.))
    sigma_gamma =  pyro.sample("sigma_gamma", dist.HalfCauchy(5.))
    with pyro.plate('alpha_', J):
        alpha =  pyro.sample("alpha", dist.Normal(0., sigma_alpha))
    with pyro.plate('beta_', K):
        beta =  pyro.sample("beta", dist.Normal(0., sigma_beta))
        log_gamma =  pyro.sample("log_gamma", dist.Normal(0., sigma_gamma))
    delta =  pyro.sample("delta", dist.Cauchy(0., 5))
    with pyro.plate('data', N):
        y = pyro.sample('y', dist.Bernoulli(logits= log_gamma[kk].exp()
                                            * (alpha[jj] - beta[kk] + delta)), obs=y)
