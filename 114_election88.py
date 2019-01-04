# model file: ../example-models/ARM/Ch.19/election88.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_age' in data, 'variable not found in data: key=n_age'
    assert 'n_age_edu' in data, 'variable not found in data: key=n_age_edu'
    assert 'n_edu' in data, 'variable not found in data: key=n_edu'
    assert 'n_region' in data, 'variable not found in data: key=n_region'
    assert 'n_state' in data, 'variable not found in data: key=n_state'
    assert 'age' in data, 'variable not found in data: key=age'
    assert 'age_edu' in data, 'variable not found in data: key=age_edu'
    assert 'black' in data, 'variable not found in data: key=black'
    assert 'edu' in data, 'variable not found in data: key=edu'
    assert 'female' in data, 'variable not found in data: key=female'
    assert 'region' in data, 'variable not found in data: key=region'
    assert 'state' in data, 'variable not found in data: key=state'
    assert 'v_prev' in data, 'variable not found in data: key=v_prev'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_age_edu = data["n_age_edu"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    age = data["age"]
    age_edu = data["age_edu"]
    black = data["black"]
    edu = data["edu"]
    female = data["female"]
    region = data["region"]
    state = data["state"]
    v_prev = data["v_prev"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_age_edu = data["n_age_edu"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    age = data["age"]
    age_edu = data["age_edu"]
    black = data["black"]
    edu = data["edu"]
    female = data["female"]
    region = data["region"]
    state = data["state"]
    v_prev = data["v_prev"]
    y = data["y"]
    # assign init values for parameters
    params["b_age"] = init_vector("b_age", dims=(n_age)) # vector
    params["b_age_edu"] = init_vector("b_age_edu", dims=(n_age_edu)) # vector
    params["b_edu"] = init_vector("b_edu", dims=(n_edu)) # vector
    params["b_region"] = init_vector("b_region", dims=(n_region)) # vector
    params["b_state"] = init_vector("b_state", dims=(n_state)) # vector
    params["b_v_prev"] = init_real("b_v_prev") # real/double
    params["beta"] = init_vector("beta", dims=(4)) # vector
    params["mu"] = init_real("mu") # real/double
    params["mu_age"] = init_real("mu_age") # real/double
    params["mu_age_edu"] = init_real("mu_age_edu") # real/double
    params["mu_edu"] = init_real("mu_edu") # real/double
    params["mu_region"] = init_real("mu_region") # real/double
    params["sigma_age"] = init_real("sigma_age", low=0, high=100) # real/double
    params["sigma_edu"] = init_real("sigma_edu", low=0, high=100) # real/double
    params["sigma_age_edu"] = init_real("sigma_age_edu", low=0, high=100) # real/double
    params["sigma_region"] = init_real("sigma_region", low=0, high=100) # real/double
    params["sigma_state"] = init_real("sigma_state", low=0, high=100) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_age_edu = data["n_age_edu"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    age = data["age"]
    age_edu = data["age_edu"]
    black = data["black"]
    edu = data["edu"]
    female = data["female"]
    region = data["region"]
    state = data["state"]
    v_prev = data["v_prev"]
    y = data["y"]
    # INIT parameters
    b_age = params["b_age"]
    b_age_edu = params["b_age_edu"]
    b_edu = params["b_edu"]
    b_region = params["b_region"]
    b_state = params["b_state"]
    b_v_prev = params["b_v_prev"]
    beta = params["beta"]
    mu = params["mu"]
    mu_age = params["mu_age"]
    mu_age_edu = params["mu_age_edu"]
    mu_edu = params["mu_edu"]
    mu_region = params["mu_region"]
    sigma_age = params["sigma_age"]
    sigma_edu = params["sigma_edu"]
    sigma_age_edu = params["sigma_age_edu"]
    sigma_region = params["sigma_region"]
    sigma_state = params["sigma_state"]
    # initialize transformed parameters
    b_age_adj = init_vector("b_age_adj", dims=(n_age)) # vector
    b_age_edu_adj = init_vector("b_age_edu_adj", dims=(n_age_edu)) # vector
    b_edu_adj = init_vector("b_edu_adj", dims=(n_edu)) # vector
    b_region_adj = init_vector("b_region_adj", dims=(n_region)) # vector
    b_state_hat = init_vector("b_state_hat", dims=(n_state)) # vector
    mu_adj = init_real("mu_adj") # real/double
    Xbeta = init_vector("Xbeta", dims=(N)) # vector
    p = init_vector("p", dims=(N)) # vector
    p_bound = init_vector("p_bound", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        Xbeta[i - 1] = _pyro_assign(Xbeta[i - 1], (((((((_index_select(beta, 1 - 1)  + (_index_select(beta, 2 - 1)  * _index_select(female, i - 1) )) + (_index_select(beta, 3 - 1)  * _index_select(black, i - 1) )) + ((_index_select(beta, 4 - 1)  * _index_select(female, i - 1) ) * _index_select(black, i - 1) )) + _index_select(b_age, age[i - 1] - 1) ) + _index_select(b_edu, edu[i - 1] - 1) ) + _index_select(b_age_edu, age_edu[i - 1] - 1) ) + _index_select(b_state, state[i - 1] - 1) ))
    mu_adj = _pyro_assign(mu_adj, ((((_index_select(beta, 1 - 1)  + _call_func("mean", [b_age])) + _call_func("mean", [b_edu])) + _call_func("mean", [b_age_edu])) + _call_func("mean", [b_state])))
    b_age_adj = _pyro_assign(b_age_adj, _call_func("subtract", [b_age,_call_func("mean", [b_age])]))
    b_edu_adj = _pyro_assign(b_edu_adj, _call_func("subtract", [b_edu,_call_func("mean", [b_edu])]))
    b_age_edu_adj = _pyro_assign(b_age_edu_adj, _call_func("subtract", [b_age_edu,_call_func("mean", [b_age_edu])]))
    b_region_adj = _pyro_assign(b_region_adj, _call_func("subtract", [b_region,_call_func("mean", [b_region])]))
    for j in range(1, to_int(n_state) + 1):
        b_state_hat[j - 1] = _pyro_assign(b_state_hat[j - 1], (_index_select(b_region, region[j - 1] - 1)  + ((100 * b_v_prev) * _index_select(v_prev, j - 1) )))
    # model block

    mu_age =  _pyro_sample(mu_age, "mu_age", "normal", [0, 1])
    mu_edu =  _pyro_sample(mu_edu, "mu_edu", "normal", [0, 1])
    mu_age_edu =  _pyro_sample(mu_age_edu, "mu_age_edu", "normal", [0, 1])
    mu_region =  _pyro_sample(mu_region, "mu_region", "normal", [0, 1])
    mu =  _pyro_sample(mu, "mu", "normal", [0, 100])
    sigma_age =  _pyro_sample(sigma_age, "sigma_age", "uniform", [0, 100])
    sigma_edu =  _pyro_sample(sigma_edu, "sigma_edu", "uniform", [0, 100])
    sigma_age_edu =  _pyro_sample(sigma_age_edu, "sigma_age_edu", "uniform", [0, 100])
    sigma_region =  _pyro_sample(sigma_region, "sigma_region", "uniform", [0, 100])
    sigma_state =  _pyro_sample(sigma_state, "sigma_state", "uniform", [0, 100])
    beta =  _pyro_sample(beta, "beta", "normal", [0, 100])
    b_age =  _pyro_sample(b_age, "b_age", "normal", [(100 * mu_age), sigma_age])
    b_edu =  _pyro_sample(b_edu, "b_edu", "normal", [(100 * mu_edu), sigma_edu])
    b_age_edu =  _pyro_sample(b_age_edu, "b_age_edu", "normal", [(100 * mu_age_edu), sigma_age_edu])
    b_state =  _pyro_sample(b_state, "b_state", "normal", [b_state_hat, sigma_state])
    b_v_prev =  _pyro_sample(b_v_prev, "b_v_prev", "normal", [0, 1])
    b_region =  _pyro_sample(b_region, "b_region", "normal", [(100 * mu_region), sigma_region])
    y =  _pyro_sample(y, "y", "bernoulli_logit", [Xbeta], obs=y)

