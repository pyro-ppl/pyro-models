# model file: ../example-models/ARM/Ch.13/pilots.stan
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

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    group_id = data["group_id"]
    scenario_id = data["scenario_id"]
    y = data["y"]
    # assign init values for parameters
    params["gamma"] = init_vector("gamma", dims=(n_groups)) # vector
    params["delta"] = init_vector("delta", dims=(n_scenarios)) # vector
    params["mu"] = pyro.sample("mu"))
    params["sigma_gamma"] = pyro.sample("sigma_gamma", dist.Uniform(0., 100.))
    params["sigma_delta"] = pyro.sample("sigma_delta", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

def model(data, params):
    # initialize data
    N = data["N"]
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    group_id = data["group_id"]
    scenario_id = data["scenario_id"]
    y = data["y"]
    
    # init parameters
    gamma = params["gamma"]
    delta = params["delta"]
    mu = params["mu"]
    sigma_gamma = params["sigma_gamma"]
    sigma_delta = params["sigma_delta"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], ((mu + _index_select(gamma, group_id[i - 1] - 1) ) + _index_select(delta, scenario_id[i - 1] - 1) ))
    # model block

    gamma =  _pyro_sample(gamma, "gamma", "normal", [0., sigma_gamma])
    delta =  _pyro_sample(delta, "delta", "normal", [0., sigma_delta])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

