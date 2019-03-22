# model file: example-models/ARM/Ch.4/mesquite_log.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'weight' in data, 'variable not found in data: key=weight'
    assert 'diam1' in data, 'variable not found in data: key=diam1'
    assert 'diam2' in data, 'variable not found in data: key=diam2'
    assert 'canopy_height' in data, 'variable not found in data: key=canopy_height'
    assert 'total_height' in data, 'variable not found in data: key=total_height'
    assert 'density' in data, 'variable not found in data: key=density'
    assert 'group' in data, 'variable not found in data: key=group'
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    total_height = data["total_height"]
    density = data["density"]
    group = data["group"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    total_height = data["total_height"]
    density = data["density"]
    group = data["group"]
    log_weight = torch.log(weight)
    log_diam1 = torch.log(diam1)
    log_diam2 = torch.log(diam2)
    log_canopy_height = torch.log(canopy_height)
    log_total_height  = torch.log(total_height)
    log_density       = torch.log(density)
    data["log_weight"] = log_weight
    data["log_diam1"] = log_diam1
    data["log_diam2"] = log_diam2
    data["log_canopy_height"] = log_canopy_height
    data["log_total_height"] = log_total_height
    data["log_density"] = log_density

def init_params(data):
    params = {}
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(7)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    weight = data["weight"]
    diam1 = data["diam1"]
    diam2 = data["diam2"]
    canopy_height = data["canopy_height"]
    total_height = data["total_height"]
    density = data["density"]
    group = data["group"]
    # initialize transformed data
    log_weight = data["log_weight"]
    log_diam1 = data["log_diam1"]
    log_diam2 = data["log_diam2"]
    log_canopy_height = data["log_canopy_height"]
    log_total_height = data["log_total_height"]
    log_density = data["log_density"]

    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    with pyro.plate("data", N):
        pyro.sample('log_weight', dist.Normal(beta[0] + beta[1] * log_diam1 + beta[2] * log_diam2
                                          + beta[3] * log_canopy_height + beta[4] * log_total_height
                                          + beta[5] * log_density + beta[6] * group, sigma),
                    obs=log_weight)
