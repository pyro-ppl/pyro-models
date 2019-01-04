# model file: ../example-models/ARM/Ch.14/election88.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_state' in data, 'variable not found in data: key=n_state'
    assert 'black' in data, 'variable not found in data: key=black'
    assert 'female' in data, 'variable not found in data: key=female'
    assert 'state' in data, 'variable not found in data: key=state'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_state = data["n_state"]
    black = data["black"]
    female = data["female"]
    state = data["state"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_state = data["n_state"]
    black = data["black"]
    female = data["female"]
    state = data["state"]
    y = data["y"]
    # assign init values for parameters
    params["a"] = init_vector("a", dims=(n_state)) # vector
    params["b"] = init_vector("b", dims=(2)) # vector
    params["sigma_a"] = init_real("sigma_a", low=0, high=100) # real/double
    params["mu_a"] = init_real("mu_a") # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n_state = data["n_state"]
    black = data["black"]
    female = data["female"]
    state = data["state"]
    y = data["y"]
    # INIT parameters
    a = params["a"]
    b = params["b"]
    sigma_a = params["sigma_a"]
    mu_a = params["mu_a"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (((_index_select(b, 1 - 1)  * _index_select(black, i - 1) ) + (_index_select(b, 2 - 1)  * _index_select(female, i - 1) )) + _index_select(a, state[i - 1] - 1) ))
    # model block

    mu_a =  _pyro_sample(mu_a, "mu_a", "normal", [0, 1])
    a =  _pyro_sample(a, "a", "normal", [mu_a, sigma_a])
    b =  _pyro_sample(b, "b", "normal", [0, 100])
    y =  _pyro_sample(y, "y", "bernoulli_logit", [y_hat], obs=y)

