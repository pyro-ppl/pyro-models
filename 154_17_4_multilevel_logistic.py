# model file: ../example-models/ARM/Ch.17/17.4_multilevel_logistic.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_age' in data, 'variable not found in data: key=n_age'
    assert 'n_edu' in data, 'variable not found in data: key=n_edu'
    assert 'n_region' in data, 'variable not found in data: key=n_region'
    assert 'n_state' in data, 'variable not found in data: key=n_state'
    assert 'female' in data, 'variable not found in data: key=female'
    assert 'black' in data, 'variable not found in data: key=black'
    assert 'age' in data, 'variable not found in data: key=age'
    assert 'edu' in data, 'variable not found in data: key=edu'
    assert 'region' in data, 'variable not found in data: key=region'
    assert 'state' in data, 'variable not found in data: key=state'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'v_prev' in data, 'variable not found in data: key=v_prev'
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    female = data["female"]
    black = data["black"]
    age = data["age"]
    edu = data["edu"]
    region = data["region"]
    state = data["state"]
    y = data["y"]
    v_prev = data["v_prev"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    female = data["female"]
    black = data["black"]
    age = data["age"]
    edu = data["edu"]
    region = data["region"]
    state = data["state"]
    y = data["y"]
    v_prev = data["v_prev"]
    # assign init values for parameters
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))
    params["sigma_age"] = pyro.sample("sigma_age", dist.Uniform(0))
    params["sigma_edu"] = pyro.sample("sigma_edu", dist.Uniform(0))
    params["sigma_state"] = pyro.sample("sigma_state", dist.Uniform(0))
    params["sigma_region"] = pyro.sample("sigma_region", dist.Uniform(0))
    params["sigma_age_edu"] = pyro.sample("sigma_age_edu", dist.Uniform(0))
    params["b_0"] = pyro.sample("b_0"))
    params["b_female"] = pyro.sample("b_female"))
    params["b_black"] = pyro.sample("b_black"))
    params["b_female_black"] = pyro.sample("b_female_black"))
    params["b_v_prev"] = pyro.sample("b_v_prev"))
    params["b_age"] = init_vector("b_age", dims=(n_age)) # vector
    params["b_edu"] = init_vector("b_edu", dims=(n_edu)) # vector
    params["b_region"] = init_vector("b_region", dims=(n_region)) # vector
    params["b_age_edu"] = init_matrix("b_age_edu", dims=(n_age, n_edu)) # matrix
    params["b_hat"] = init_vector("b_hat", dims=(n_state)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    female = data["female"]
    black = data["black"]
    age = data["age"]
    edu = data["edu"]
    region = data["region"]
    state = data["state"]
    y = data["y"]
    v_prev = data["v_prev"]
    
    # init parameters
    sigma = params["sigma"]
    sigma_age = params["sigma_age"]
    sigma_edu = params["sigma_edu"]
    sigma_state = params["sigma_state"]
    sigma_region = params["sigma_region"]
    sigma_age_edu = params["sigma_age_edu"]
    b_0 = params["b_0"]
    b_female = params["b_female"]
    b_black = params["b_black"]
    b_female_black = params["b_female_black"]
    b_v_prev = params["b_v_prev"]
    b_age = params["b_age"]
    b_edu = params["b_edu"]
    b_region = params["b_region"]
    b_age_edu = params["b_age_edu"]
    b_hat = params["b_hat"]
    # initialize transformed parameters
    # model block
    # {
    p = init_vector("p", dims=(N)) # vector
    b_state_hat = init_vector("b_state_hat", dims=(n_state)) # vector

    b_0 =  _pyro_sample(b_0., "b_0", "normal", [0., 100])
    b_female =  _pyro_sample(b_female, "b_female", "normal", [0., 100])
    b_black =  _pyro_sample(b_black, "b_black", "normal", [0., 100])
    b_female_black =  _pyro_sample(b_female_black, "b_female_black", "normal", [0., 100])
    b_age =  _pyro_sample(b_age, "b_age", "normal", [0., sigma_age])
    b_edu =  _pyro_sample(b_edu, "b_edu", "normal", [0., sigma_edu])
    b_region =  _pyro_sample(b_region, "b_region", "normal", [0., sigma_region])
    for j in range(1, to_int(n_age) + 1):

        for i in range(1, to_int(n_edu) + 1):
            b_age_edu[j - 1][i - 1] =  _pyro_sample(_index_select(_index_select(b_age_edu, j - 1) , i - 1) , "b_age_edu[%d][%d]" % (to_int(j-1),to_int(i-1)), "normal", [0., sigma_age_edu])
    b_v_prev =  _pyro_sample(b_v_prev, "b_v_prev", "normal", [0., 100])
    for j in range(1, to_int(n_state) + 1):
        b_state_hat[j - 1] = _pyro_assign(b_state_hat[j - 1], (_index_select(b_region, region[j - 1] - 1)  + (b_v_prev * _index_select(v_prev, j - 1) )))
    b_hat =  _pyro_sample(b_hat, "b_hat", "normal", [b_state_hat, sigma_state])
    for i in range(1, to_int(N) + 1):
        p[i - 1] = _pyro_assign(p[i - 1], _call_func("fmax", [0.,_call_func("fmin", [1,_call_func("inv_logit", [(((((((b_0 + (b_female * _index_select(female, i - 1) )) + (b_black * _index_select(black, i - 1) )) + ((b_female_black * _index_select(female, i - 1) ) * _index_select(black, i - 1) )) + _index_select(b_age, age[i - 1] - 1) ) + _index_select(b_edu, edu[i - 1] - 1) ) + _index_select(_index_select(b_age_edu, age[i - 1] - 1) , edu[i - 1] - 1) ) + _index_select(b_hat, state[i - 1] - 1) )])])]))
    y =  _pyro_sample(y, "y", "bernoulli", [p], obs=y)
    # }

