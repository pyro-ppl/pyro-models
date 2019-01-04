# model file: ../example-models/ARM/Ch.23/electric.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_pair' in data, 'variable not found in data: key=n_pair'
    assert 'pair' in data, 'variable not found in data: key=pair'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    pair = data["pair"]
    treatment = data["treatment"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    pair = data["pair"]
    treatment = data["treatment"]
    y = data["y"]
    # assign init values for parameters
    params["a"] = init_vector("a", dims=(n_pair)) # vector
    params["beta"] = init_real("beta") # real/double
    params["mu_a"] = init_real("mu_a") # real/double
    params["sigma_a"] = init_real("sigma_a", low=0, high=100) # real/double
    params["sigma_y"] = init_real("sigma_y", low=0, high=100) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    pair = data["pair"]
    treatment = data["treatment"]
    y = data["y"]
    # INIT parameters
    a = params["a"]
    beta = params["beta"]
    mu_a = params["mu_a"]
    sigma_a = params["sigma_a"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (_index_select(a, pair[i - 1] - 1)  + (beta * _index_select(treatment, i - 1) )))
    # model block

    mu_a =  _pyro_sample(mu_a, "mu_a", "normal", [0, 1])
    sigma_a =  _pyro_sample(sigma_a, "sigma_a", "uniform", [0, 100])
    sigma_y =  _pyro_sample(sigma_y, "sigma_y", "uniform", [0, 100])
    a =  _pyro_sample(a, "a", "normal", [(100 * mu_a), sigma_a])
    beta =  _pyro_sample(beta, "beta", "normal", [0, 1])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

