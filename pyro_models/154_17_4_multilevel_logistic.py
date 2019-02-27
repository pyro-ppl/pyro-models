# model file: example-models/ARM/Ch.17/17.4_multilevel_logistic.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def inv_logit(x):
    return x.exp() / (1 + x.exp())

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
    params["sigma"] = pyro.sample("sigma", dist.HalfCauchy(2.5))
    params["sigma_age"] = pyro.sample("sigma_age", dist.HalfCauchy(2.5))
    params["sigma_edu"] = pyro.sample("sigma_edu", dist.HalfCauchy(2.5))
    params["sigma_state"] = pyro.sample("sigma_state", dist.HalfCauchy(2.5))
    params["sigma_region"] = pyro.sample("sigma_region", dist.HalfCauchy(2.5))
    params["sigma_age_edu"] = pyro.sample("sigma_age_edu", dist.HalfCauchy(2.5))
    return params

def model(data, params):
    # XXX NaN issues
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    female = data["female"]
    black = data["black"]
    age = data["age"].long() - 1
    edu = data["edu"].long() - 1
    region = data["region"].long() - 1
    state = data["state"].long() - 1
    y = data["y"]
    v_prev = data["v_prev"]

    # init parameters
    sigma = params["sigma"]
    sigma_age = params["sigma_age"]
    sigma_edu = params["sigma_edu"]
    sigma_state = params["sigma_state"]
    sigma_region = params["sigma_region"]
    sigma_age_edu = params["sigma_age_edu"]

    b_0 =  pyro.sample("b_0", dist.Normal(0., 100.))
    b_female =  pyro.sample("b_female", dist.Normal(0., 100.))
    b_black =  pyro.sample("b_black", dist.Normal(0., 100.))
    b_female_black =  pyro.sample("b_female_black", dist.Normal(0., 100.))
    b_v_prev =  pyro.sample("b_p_prev", dist.Normal(0., 100.))

    n_age_plate = pyro.plate("n_age", n_age, dim=-2)
    n_edu_plate = pyro.plate("n_edu", n_edu, dim=-1)
    with n_age_plate:
        b_age =  pyro.sample("b_age", dist.Normal(0., sigma_age))
    with n_edu_plate:
        b_edu =  pyro.sample("b_edu", dist.Normal(0., sigma_edu))
    with pyro.plate("n_region", n_region):
        b_region =  pyro.sample("b_region", dist.Normal(0., sigma_region))
    with pyro.plate('n_state', n_state):
        b_state_hat = b_region[region] + b_v_prev * v_prev
        b_hat = pyro.sample('b_hat', dist.Normal(b_state_hat, sigma_state))
    with n_age_plate, n_edu_plate:
        b_age_edu = pyro.sample('b_age_edu', dist.Normal(0., sigma_age_edu))
    with pyro.plate("data", N):
        p = inv_logit(b_0 + b_female*female
                      + b_black*black + b_female_black*female*black +
                      b_age[age].squeeze(1) + b_edu[edu] + b_age_edu[age,edu] +
                      b_hat[state]).clamp(min=1e-6, max=1-1e-6)
        pyro.sample('y', dist.Bernoulli(p), obs=y)
