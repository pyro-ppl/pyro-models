# model file: ../example-models/ARM/Ch.23/electric_1b.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



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
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    pair = data["pair"]
    pre_test = data["pre_test"]
    treatment = data["treatment"]
    y = data["y"]
    # assign init values for parameters
    params["a"] = init_vector("a", dims=(n_pair)) # vector
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["mu_a"] = pyro.sample("mu_a"))
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    pair = data["pair"]
    pre_test = data["pre_test"]
    treatment = data["treatment"]
    y = data["y"]
    
    # init parameters
    a = params["a"]
    beta = params["beta"]
    mu_a = params["mu_a"]
    sigma_a = params["sigma_a"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], ((_index_select(a, pair[i - 1] - 1)  + (_index_select(beta, 1 - 1)  * _index_select(treatment, i - 1) )) + (_index_select(beta, 2 - 1)  * _index_select(pre_test, i - 1) )))
    # model block

    mu_a =  _pyro_sample(mu_a, "mu_a", "normal", [0., 1])
    a =  _pyro_sample(a, "a", "normal", [(100 * mu_a), sigma_a])
    beta =  _pyro_sample(beta, "beta", "normal", [0., 100])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

