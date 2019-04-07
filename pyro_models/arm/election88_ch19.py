# model file: example-models/ARM/Ch.19/election88.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



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

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_age_edu = data["n_age_edu"]
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    age = data["age"].long() - 1
    age_edu = data["age_edu"].long() - 1
    black = data["black"]
    edu = data["edu"].long() - 1
    female = data["female"]
    region = data["region"].long() - 1
    state = data["state"].long() - 1
    v_prev = data["v_prev"]
    y = data["y"]

    # model block
    mu_age =  pyro.sample("mu_age", dist.Normal(0., 1.))
    mu_edu =  pyro.sample("mu_edu", dist.Normal(0., 1.))
    mu_age_edu =  pyro.sample("mu_age_edu", dist.Normal(0., 1.))
    mu_region =  pyro.sample("mu_region", dist.Normal(0., 1.))
    mu =  pyro.sample("mu", dist.Normal(0., 100.))
    sigma_age =  pyro.sample("sigma_age", dist.Uniform(0., 100.))
    sigma_edu =  pyro.sample("sigma_edu", dist.Uniform(0., 100.))
    sigma_age_edu =  pyro.sample("sigma_age_edu", dist.Uniform(0., 100.))
    sigma_region =  pyro.sample("sigma_region", dist.Uniform(0., 100.))
    sigma_state =  pyro.sample("sigma_state", dist.Uniform(0., 100.))
    with pyro.plate('beta_data', 4):
        beta =  pyro.sample("beta", dist.Normal(0., 100.))
    with pyro.plate('b_age_data', n_age):
        b_age =  pyro.sample("b_age", dist.Normal((100 * mu_age), sigma_age))
    with pyro.plate('b_edu_data', n_edu):
        b_edu =  pyro.sample("b_edu", dist.Normal((100 * mu_edu), sigma_edu))
    with pyro.plate('b_age_edu_data', n_age_edu):
        b_age_edu =  pyro.sample("b_age_edu", dist.Normal((100 * mu_age_edu), sigma_age_edu))
    with pyro.plate('region_data', n_region):
        b_region =  pyro.sample("b_region", dist.Normal((100 * mu_region), sigma_region))
    b_v_prev =  pyro.sample("b_v_prev", dist.Normal(0., 1.))
    with pyro.plate('state', n_state):
        b_state_hat = b_region[...,region] + 100 * b_v_prev * v_prev
        b_state =  pyro.sample("b_state", dist.Normal(b_state_hat, sigma_state))
    with pyro.plate('data', N):
        Xbeta = beta[...,0].unsqueeze(-1) + beta[...,1].unsqueeze(-1)*female + beta[...,2].unsqueeze(-1) * black + beta[...,3].unsqueeze(-1) * female * black + \
                b_age[...,age] + b_edu[...,edu] + b_age_edu[...,age_edu] + b_state[...,state]
        y =  pyro.sample("y", dist.Bernoulli(logits=Xbeta), obs=y)
