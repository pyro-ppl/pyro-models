# model file: ../example-models/ARM/Ch.14/pilots.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



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
    # initialize data
    N = data["N"]
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    group_id = data["group_id"]
    scenario_id = data["scenario_id"]
    y = data["y"]
    # assign init values for parameters
    params["a"] = init_vector("a", dims=(n_groups)) # vector
    params["b"] = init_vector("b", dims=(n_scenarios)) # vector
    params["mu_a"] = pyro.sample("mu_a"))
    params["mu_b"] = pyro.sample("mu_b"))
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    params["sigma_b"] = pyro.sample("sigma_b", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    group_id = data["group_id"]
    scenario_id = data["scenario_id"]
    y = data["y"]
    
    # init parameters
    a = params["a"]
    b = params["b"]
    mu_a = params["mu_a"]
    mu_b = params["mu_b"]
    sigma_a = params["sigma_a"]
    sigma_b = params["sigma_b"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (_index_select(a, group_id[i - 1] - 1)  + _index_select(b, scenario_id[i - 1] - 1) ))
    # model block

    mu_a =  _pyro_sample(mu_a, "mu_a", "normal", [0., 1])
    a =  _pyro_sample(a, "a", "normal", [(10 * mu_a), sigma_a])
    mu_b =  _pyro_sample(mu_b, "mu_b", "normal", [0., 1])
    b =  _pyro_sample(b, "b", "normal", [(10 * mu_b), sigma_b])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

