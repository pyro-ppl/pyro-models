# model file: ../example-models/ARM/Ch.13/earnings_latin_square_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_age' in data, 'variable not found in data: key=n_age'
    assert 'n_eth' in data, 'variable not found in data: key=n_eth'
    assert 'age' in data, 'variable not found in data: key=age'
    assert 'eth' in data, 'variable not found in data: key=eth'
    assert 'x_centered' in data, 'variable not found in data: key=x_centered'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"]
    eth = data["eth"]
    x_centered = data["x_centered"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"]
    eth = data["eth"]
    x_centered = data["x_centered"]
    y = data["y"]
    # assign init values for parameters
    params["eta_a1"] = init_vector("eta_a1", dims=(n_eth)) # vector
    params["eta_a2"] = init_vector("eta_a2", dims=(n_eth)) # vector
    params["eta_b1"] = init_vector("eta_b1", dims=(n_age)) # vector
    params["eta_b2"] = init_vector("eta_b2", dims=(n_age)) # vector
    params["eta_c"] = init_matrix("eta_c", dims=(n_eth, n_age)) # matrix
    params["eta_d"] = init_matrix("eta_d", dims=(n_eth, n_age)) # matrix
    params["mu_a1"] = pyro.sample("mu_a1"))
    params["mu_a2"] = pyro.sample("mu_a2"))
    params["mu_b1"] = pyro.sample("mu_b1"))
    params["mu_b2"] = pyro.sample("mu_b2"))
    params["mu_c"] = pyro.sample("mu_c"))
    params["mu_d"] = pyro.sample("mu_d"))
    params["sigma_a1"] = pyro.sample("sigma_a1", dist.Uniform(0., 100.))
    params["sigma_a2"] = pyro.sample("sigma_a2", dist.Uniform(0., 100.))
    params["sigma_b1"] = pyro.sample("sigma_b1", dist.Uniform(0., 100.))
    params["sigma_b2"] = pyro.sample("sigma_b2", dist.Uniform(0., 100.))
    params["sigma_c"] = pyro.sample("sigma_c", dist.Uniform(0., 100.))
    params["sigma_d"] = pyro.sample("sigma_d", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

def model(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"]
    eth = data["eth"]
    x_centered = data["x_centered"]
    y = data["y"]
    
    # init parameters
    eta_a1 = params["eta_a1"]
    eta_a2 = params["eta_a2"]
    eta_b1 = params["eta_b1"]
    eta_b2 = params["eta_b2"]
    eta_c = params["eta_c"]
    eta_d = params["eta_d"]
    mu_a1 = params["mu_a1"]
    mu_a2 = params["mu_a2"]
    mu_b1 = params["mu_b1"]
    mu_b2 = params["mu_b2"]
    mu_c = params["mu_c"]
    mu_d = params["mu_d"]
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_b1 = params["sigma_b1"]
    sigma_b2 = params["sigma_b2"]
    sigma_c = params["sigma_c"]
    sigma_d = params["sigma_d"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    a1 = init_vector("a1", dims=(n_eth)) # vector
    a2 = init_vector("a2", dims=(n_eth)) # vector
    b1 = init_vector("b1", dims=(n_age)) # vector
    b2 = init_vector("b2", dims=(n_age)) # vector
    c = init_matrix("c", dims=(n_eth, n_age)) # matrix
    d = init_matrix("d", dims=(n_eth, n_age)) # matrix
    y_hat = init_vector("y_hat", dims=(N)) # vector
    a1 = _pyro_assign(a1, _call_func("add", [(5 * mu_a1),_call_func("multiply", [sigma_a1,eta_a1])]))
    a2 = _pyro_assign(a2, _call_func("add", [mu_a2,_call_func("multiply", [sigma_a2,eta_a2])]))
    b1 = _pyro_assign(b1, _call_func("add", [(5 * mu_b1),_call_func("multiply", [sigma_b1,eta_b1])]))
    b2 = _pyro_assign(b2, _call_func("add", [(0.10000000000000001 * mu_b2),_call_func("multiply", [sigma_b2,eta_b2])]))
    c = _pyro_assign(c, _call_func("add", [(0.10000000000000001 * mu_c),_call_func("multiply", [sigma_c,eta_c])]))
    d = _pyro_assign(d, _call_func("add", [(0.01 * mu_d),_call_func("multiply", [sigma_d,eta_d])]))
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (((((_index_select(a1, eth[i - 1] - 1)  + (_index_select(a2, eth[i - 1] - 1)  * _index_select(x_centered, i - 1) )) + _index_select(b1, age[i - 1] - 1) ) + (_index_select(b2, age[i - 1] - 1)  * _index_select(x_centered, i - 1) )) + _index_select(_index_select(c, eth[i - 1] - 1) , age[i - 1] - 1) ) + (_index_select(_index_select(d, eth[i - 1] - 1) , age[i - 1] - 1)  * _index_select(x_centered, i - 1) )))
    # model block

    mu_a1 =  _pyro_sample(mu_a1, "mu_a1", "normal", [0., 1])
    mu_a2 =  _pyro_sample(mu_a2, "mu_a2", "normal", [0., 1])
    eta_a1 =  _pyro_sample(eta_a1, "eta_a1", "normal", [0., 1])
    eta_a2 =  _pyro_sample(eta_a2, "eta_a2", "normal", [0., 1])
    mu_b1 =  _pyro_sample(mu_b1, "mu_b1", "normal", [0., 1])
    mu_b2 =  _pyro_sample(mu_b2, "mu_b2", "normal", [0., 1])
    eta_b1 =  _pyro_sample(eta_b1, "eta_b1", "normal", [0., 1])
    eta_b2 =  _pyro_sample(eta_b2, "eta_b2", "normal", [0., 1])
    mu_c =  _pyro_sample(mu_c, "mu_c", "normal", [0., 1])
    for j in range(1, to_int(n_eth) + 1):
        eta_c[j - 1] =  _pyro_sample(_index_select(eta_c, j - 1) , "eta_c[%d]" % (to_int(j-1)), "normal", [0., 1])
    mu_d =  _pyro_sample(mu_d, "mu_d", "normal", [0., 1])
    for j in range(1, to_int(n_eth) + 1):
        eta_d[j - 1] =  _pyro_sample(_index_select(eta_d, j - 1) , "eta_d[%d]" % (to_int(j-1)), "normal", [0., 1])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

