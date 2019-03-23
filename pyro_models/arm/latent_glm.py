# model file: example-models/ARM/Ch.19/election88.stan
import torch
import pyro
import pyro.distributions as dist


def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def inv_logit(x):
    return 1. / (1. + torch.exp(-x))

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
    n_edu = data["n_edu"]
    n_region = data["n_region"]
    n_state = data["n_state"]
    black = data["black"]
    female = data["female"]
    age = data["age"].long() - 1
    edu = data["edu"].long() - 1
    region = data["region"].long() - 1
    state = data["state"].long() - 1
    v_prev = data["v_prev"]
    y = data["y"]
#     z = data["z"]

    # define plates
    age_plate = pyro.plate('age_p', n_age, dim=-2)
    edu_plate = pyro.plate('edu_p', n_edu, dim=-1)

    # initialize parameters
    sigma =  pyro.sample("sigma", dist.HalfCauchy(2.5))
    sigma_age =  pyro.sample("sigma_age", dist.HalfCauchy(2.5))
    sigma_edu =  pyro.sample("sigma_edu", dist.HalfCauchy(2.5))
    sigma_state =  pyro.sample("sigma_state", dist.HalfCauchy(2.5))
    sigma_region =  pyro.sample("sigma_region", dist.HalfCauchy(2.5))
    sigma_age_edu =  pyro.sample("sigma_age_edu", dist.HalfCauchy(2.5))

    # model block
    b_0 =  pyro.sample("b_0", dist.Normal(0., 100.))
    b_female =  pyro.sample("b_female", dist.Normal(0., 100.))
    b_black =  pyro.sample("b_black", dist.Normal(0., 100.))
    b_female_black =  pyro.sample("b_female_black", dist.Normal(0., 100.))
    b_v_prev =  pyro.sample("b_v_prev", dist.Normal(0., 100.))

    with age_plate:
        b_age =  pyro.sample("b_age", dist.Normal(0., sigma_age))
    with edu_plate:
        b_edu  =  pyro.sample("b_edu", dist.Normal(0., sigma_edu))
    with pyro.plate('region_p', n_region):
        b_region =  pyro.sample("b_region", dist.Normal(0., sigma_region))
    with age_plate, edu_plate:
        b_age_edu =  pyro.sample("b_age_edu", dist.Normal(0., sigma_age_edu))

    with pyro.plate('state', n_state):
        b_state_hat = b_region[region] + b_v_prev * v_prev
        b_state =  pyro.sample("b_state", dist.Normal(b_state_hat, sigma_state))

    with pyro.plate('data', N):
        Xbeta = b_0 + b_female * female + b_black * black + b_female_black * female * black + \
                b_age[age].squeeze(-1) + b_edu[edu] + b_age_edu[age, edu] + b_state[state]
        p = inv_logit(Xbeta).clamp(min=0., max=1.0)
        y =  pyro.sample("y", dist.Bernoulli(p), obs=y)
#         z_lo = 100 * (y == 0)
#         z_hi = 100 * (y == 1)
          # truncated distribution
#         z =  pyro.sample("z", dist.Logistic(Xbeta, 1), obs=z)

