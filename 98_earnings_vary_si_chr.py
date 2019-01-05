# model file: ../example-models/ARM/Ch.13/earnings_vary_si_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'earn' in data, 'variable not found in data: key=earn'
    assert 'eth' in data, 'variable not found in data: key=eth'
    assert 'height' in data, 'variable not found in data: key=height'
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    log_earn = init_vector("log_earn", dims=(N)) # vector
    log_earn = _pyro_assign(log_earn, _call_func("log", [earn]))
    data["log_earn"] = log_earn

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    # initialize transformed data
    log_earn = data["log_earn"]
    # assign init values for parameters
    params["eta1"] = init_vector("eta1", dims=(4)) # vector
    params["eta2"] = init_vector("eta2", dims=(4)) # vector
    params["mu_a1"] = pyro.sample("mu_a1"))
    params["mu_a2"] = pyro.sample("mu_a2"))
    params["xi"] = pyro.sample("xi"))
    params["sigma_a1"] = pyro.sample("sigma_a1", dist.Uniform(0))
    params["sigma_a2"] = pyro.sample("sigma_a2", dist.Uniform(0))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    earn = data["earn"]
    eth = data["eth"]
    height = data["height"]
    # initialize transformed data
    log_earn = data["log_earn"]
    
    # init parameters
    eta1 = params["eta1"]
    eta2 = params["eta2"]
    mu_a1 = params["mu_a1"]
    mu_a2 = params["mu_a2"]
    xi = params["xi"]
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    a1 = init_vector("a1", dims=(4)) # vector
    a2 = init_vector("a2", dims=(4)) # vector
    y_hat = init_vector("y_hat", dims=(N)) # vector
    a1 = _pyro_assign(a1, _call_func("add", [(10 * mu_a1),_call_func("multiply", [sigma_a1,eta1])]))
    a2 = _pyro_assign(a2, _call_func("add", [(0.10000000000000001 * mu_a2),_call_func("multiply", [sigma_a2,eta2])]))
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (_index_select(a1, eth[i - 1] - 1)  + (_index_select(a2, eth[i - 1] - 1)  * _index_select(height, i - 1) )))
    # model block

    mu_a1 =  _pyro_sample(mu_a1, "mu_a1", "normal", [0., 1])
    mu_a2 =  _pyro_sample(mu_a2, "mu_a2", "normal", [0., 1])
    eta1 =  _pyro_sample(eta1, "eta1", "normal", [0., 1])
    eta2 =  _pyro_sample(eta2, "eta2", "normal", [0., 1])
    sigma_a1 =  _pyro_sample(sigma_a1, "sigma_a1", "cauchy", [0., 5])
    sigma_a2 =  _pyro_sample(sigma_a2, "sigma_a2", "cauchy", [0., 5])
    sigma_y =  _pyro_sample(sigma_y, "sigma_y", "cauchy", [0., 5])
    log_earn =  _pyro_sample(log_earn, "log_earn", "normal", [y_hat, sigma_y], obs=log_earn)

