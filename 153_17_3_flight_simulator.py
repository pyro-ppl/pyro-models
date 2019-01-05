# model file: ../example-models/ARM/Ch.17/17.3_flight_simulator.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_treatment' in data, 'variable not found in data: key=n_treatment'
    assert 'n_airport' in data, 'variable not found in data: key=n_airport'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'airport' in data, 'variable not found in data: key=airport'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_treatment = data["n_treatment"]
    n_airport = data["n_airport"]
    treatment = data["treatment"]
    airport = data["airport"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_treatment = data["n_treatment"]
    n_airport = data["n_airport"]
    treatment = data["treatment"]
    airport = data["airport"]
    y = data["y"]
    # assign init values for parameters
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))
    params["sigma_gamma"] = pyro.sample("sigma_gamma", dist.Uniform(0))
    params["sigma_delta"] = pyro.sample("sigma_delta", dist.Uniform(0))
    params["gamma"] = init_vector("gamma", dims=(n_treatment)) # vector
    params["delta"] = init_vector("delta", dims=(n_airport)) # vector
    params["mu"] = pyro.sample("mu"))

def model(data, params):
    # initialize data
    N = data["N"]
    n_treatment = data["n_treatment"]
    n_airport = data["n_airport"]
    treatment = data["treatment"]
    airport = data["airport"]
    y = data["y"]
    
    # init parameters
    sigma = params["sigma"]
    sigma_gamma = params["sigma_gamma"]
    sigma_delta = params["sigma_delta"]
    gamma = params["gamma"]
    delta = params["delta"]
    mu = params["mu"]
    # initialize transformed parameters
    # model block
    # {
    y_hat = init_vector("y_hat", dims=(N)) # vector

    sigma =  _pyro_sample(sigma, "sigma", "uniform", [0., 100])
    sigma_gamma =  _pyro_sample(sigma_gamma, "sigma_gamma", "uniform", [0., 100])
    sigma_delta =  _pyro_sample(sigma_delta, "sigma_delta", "uniform", [0., 100])
    mu =  _pyro_sample(mu, "mu", "normal", [0., 100])
    gamma =  _pyro_sample(gamma, "gamma", "normal", [0., sigma_gamma])
    delta =  _pyro_sample(delta, "delta", "normal", [0., sigma_delta])
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], ((mu + _index_select(gamma, treatment[i - 1] - 1) ) + _index_select(delta, airport[i - 1] - 1) ))
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma], obs=y)
    # }

