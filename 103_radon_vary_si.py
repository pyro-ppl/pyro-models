# model file: ../example-models/ARM/Ch.13/radon_vary_si.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'county' in data, 'variable not found in data: key=county'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    county = data["county"]
    x = data["x"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    county = data["county"]
    x = data["x"]
    y = data["y"]
    # assign init values for parameters
    params["a1"] = init_vector("a1", dims=(85)) # vector
    params["a2"] = init_vector("a2", dims=(85)) # vector
    params["mu_a1"] = init_real("mu_a1") # real/double
    params["mu_a2"] = init_real("mu_a2") # real/double
    params["sigma_a1"] = init_real("sigma_a1", low=0, high=100) # real/double
    params["sigma_a2"] = init_real("sigma_a2", low=0, high=100) # real/double
    params["sigma_y"] = init_real("sigma_y", low=0, high=100) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    county = data["county"]
    x = data["x"]
    y = data["y"]
    # INIT parameters
    a1 = params["a1"]
    a2 = params["a2"]
    mu_a1 = params["mu_a1"]
    mu_a2 = params["mu_a2"]
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (_index_select(a1, county[i - 1] - 1)  + (_index_select(a2, county[i - 1] - 1)  * _index_select(x, i - 1) )))
    # model block

    mu_a1 =  _pyro_sample(mu_a1, "mu_a1", "normal", [0, 1])
    a1 =  _pyro_sample(a1, "a1", "normal", [mu_a1, sigma_a1])
    mu_a2 =  _pyro_sample(mu_a2, "mu_a2", "normal", [0, 1])
    a2 =  _pyro_sample(a2, "a2", "normal", [(0.10000000000000001 * mu_a2), sigma_a2])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

