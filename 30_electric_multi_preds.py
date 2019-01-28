# model file: ../example-models/ARM/Ch.23/electric_multi_preds.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'pre_test' in data, 'variable not found in data: key=pre_test'
    assert 'post_test' in data, 'variable not found in data: key=post_test'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    # initialize data
    N = data["N"]
    pre_test = data["pre_test"]
    post_test = data["post_test"]
    treatment = data["treatment"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    pre_test = data["pre_test"]
    post_test = data["post_test"]
    treatment = data["treatment"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(3)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    pre_test = data["pre_test"]
    post_test = data["post_test"]
    treatment = data["treatment"]

    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block
    pyro.sample('post_test', dist.Normal(beta[0] + beta[1] * treatment + beta[2] * pre_test, sigma), obs=post_test)
