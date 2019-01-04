# model file: ../example-models/ARM/Ch.14/election88_full.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_age' in data, 'variable not found in data: key=n_age'
    assert 'n_age_edu' in data, 'variable not found in data: key=n_age_edu'
    assert 'n_edu' in data, 'variable not found in data: key=n_edu'
    assert 'n_region_full' in data, 'variable not found in data: key=n_region_full'
    assert 'n_state' in data, 'variable not found in data: key=n_state'
    assert 'age' in data, 'variable not found in data: key=age'
    assert 'age_edu' in data, 'variable not found in data: key=age_edu'
    assert 'black' in data, 'variable not found in data: key=black'
    assert 'edu' in data, 'variable not found in data: key=edu'
    assert 'female' in data, 'variable not found in data: key=female'
    assert 'region_full' in data, 'variable not found in data: key=region_full'
    assert 'state' in data, 'variable not found in data: key=state'
    assert 'v_prev_full' in data, 'variable not found in data: key=v_prev_full'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_age_edu = data["n_age_edu"]
    n_edu = data["n_edu"]
    n_region_full = data["n_region_full"]
    n_state = data["n_state"]
    age = data["age"]
    age_edu = data["age_edu"]
    black = data["black"]
    edu = data["edu"]
    female = data["female"]
    region_full = data["region_full"]
    state = data["state"]
    v_prev_full = data["v_prev_full"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_age_edu = data["n_age_edu"]
    n_edu = data["n_edu"]
    n_region_full = data["n_region_full"]
    n_state = data["n_state"]
    age = data["age"]
    age_edu = data["age_edu"]
    black = data["black"]
    edu = data["edu"]
    female = data["female"]
    region_full = data["region_full"]
    state = data["state"]
    v_prev_full = data["v_prev_full"]
    y = data["y"]
    # assign init values for parameters
    params["a"] = init_vector("a", dims=(n_age)) # vector
    params["b"] = init_vector("b", dims=(n_edu)) # vector
    params["c"] = init_vector("c", dims=(n_age_edu)) # vector
    params["d"] = init_vector("d", dims=(n_state)) # vector
    params["e"] = init_vector("e", dims=(n_region_full)) # vector
    params["beta"] = init_vector("beta", dims=(5)) # vector
    params["sigma_a"] = init_real("sigma_a", low=0, high=100) # real/double
    params["sigma_b"] = init_real("sigma_b", low=0, high=100) # real/double
    params["sigma_c"] = init_real("sigma_c", low=0, high=100) # real/double
    params["sigma_d"] = init_real("sigma_d", low=0, high=100) # real/double
    params["sigma_e"] = init_real("sigma_e", low=0, high=100) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_age_edu = data["n_age_edu"]
    n_edu = data["n_edu"]
    n_region_full = data["n_region_full"]
    n_state = data["n_state"]
    age = data["age"]
    age_edu = data["age_edu"]
    black = data["black"]
    edu = data["edu"]
    female = data["female"]
    region_full = data["region_full"]
    state = data["state"]
    v_prev_full = data["v_prev_full"]
    y = data["y"]
    # INIT parameters
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    e = params["e"]
    beta = params["beta"]
    sigma_a = params["sigma_a"]
    sigma_b = params["sigma_b"]
    sigma_c = params["sigma_c"]
    sigma_d = params["sigma_d"]
    sigma_e = params["sigma_e"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (((((((((_index_select(beta, 1 - 1)  + (_index_select(beta, 2 - 1)  * _index_select(black, i - 1) )) + (_index_select(beta, 3 - 1)  * _index_select(female, i - 1) )) + ((_index_select(beta, 5 - 1)  * _index_select(female, i - 1) ) * _index_select(black, i - 1) )) + (_index_select(beta, 4 - 1)  * _index_select(v_prev_full, i - 1) )) + _index_select(a, age[i - 1] - 1) ) + _index_select(b, edu[i - 1] - 1) ) + _index_select(c, age_edu[i - 1] - 1) ) + _index_select(d, state[i - 1] - 1) ) + _index_select(e, region_full[i - 1] - 1) ))
    # model block

    a =  _pyro_sample(a, "a", "normal", [0, sigma_a])
    b =  _pyro_sample(b, "b", "normal", [0, sigma_b])
    c =  _pyro_sample(c, "c", "normal", [0, sigma_c])
    d =  _pyro_sample(d, "d", "normal", [0, sigma_d])
    e =  _pyro_sample(e, "e", "normal", [0, sigma_e])
    beta =  _pyro_sample(beta, "beta", "normal", [0, 100])
    y =  _pyro_sample(y, "y", "bernoulli_logit", [y_hat], obs=y)

