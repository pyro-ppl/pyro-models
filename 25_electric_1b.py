# model file: example-models/ARM/Ch.23/electric_1b.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_pair' in data, 'variable not found in data: key=n_pair'
    assert 'pair' in data, 'variable not found in data: key=pair'
    assert 'pre_test' in data, 'variable not found in data: key=pre_test'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    pair = data["pair"]
    pre_test = data["pre_test"]
    treatment = data["treatment"]
    y = data["y"]

def init_params(data):
    params = {}
    # assign init values for parameters
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    pair = data["pair"].long() - 1
    pre_test = data["pre_test"]
    treatment = data["treatment"]
    y = data["y"]

    # init parameters
    sigma_a = params["sigma_a"]
    sigma_y = params["sigma_y"]

    # model block
    mu_a =  pyro.sample("mu_a", dist.Normal(0., 1.))
    with pyro.plate('n_pair', n_pair):
        a = pyro.sample('a', dist.Normal(100 * mu_a, sigma_a))
    with pyro.plate('2', 2):
        beta =  pyro.sample("beta", dist.Normal(0., 100.))
    with pyro.plate("data", N):
        y_hat = a[pair] + beta[0] * treatment + beta[1] * pre_test
        y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)
