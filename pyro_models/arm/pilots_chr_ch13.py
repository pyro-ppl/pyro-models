# model file: example-models/ARM/Ch.13/pilots_chr.stan
import torch
import pyro
import pyro.distributions as dist


def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_groups' in data, 'variable not found in data: key=n_groups'
    assert 'n_scenarios' in data, 'variable not found in data: key=n_scenarios'
    assert 'group_id' in data, 'variable not found in data: key=group_id'
    assert 'scenario_id' in data, 'variable not found in data: key=scenario_id'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    group_id = data["group_id"]
    scenario_id = data["scenario_id"]
    y = data["y"]

def init_params(data):
    params = {}
    N = data["N"]
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    group_id = data["group_id"]
    scenario_id = data["scenario_id"]
    y = data["y"]
    # assign init values for parameters
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    params["sigma_b"] = pyro.sample("sigma_b", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    group_id = data["group_id"].long() - 1
    scenario_id = data["scenario_id"].long() - 1
    y = data["y"]

    # init parameters
    sigma_a = params["sigma_a"]
    sigma_b = params["sigma_b"]
    sigma_y = params["sigma_y"]

    eta_a = init_vector("eta_a", dims=(n_groups)) # vector
    eta_b = init_vector("eta_b", dims=(n_scenarios)) # vector
    mu_a = pyro.sample("mu_a", dist.Uniform(-1., 1.))
    mu_b = pyro.sample("mu_b", dist.Uniform(-1., 1.))
    with pyro.plate("data", N):
        a = 0.1 * mu_a + eta_a * sigma_a
        b = 0.1 * mu_b + eta_b * sigma_b
        y_hat = a[group_id] + b[scenario_id]
        y =  pyro.sample("y", dist.Normal(y_hat, sigma_y), obs=y)
