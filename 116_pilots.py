# model file: ../example-models/ARM/Ch.19/pilots.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_airport' in data, 'variable not found in data: key=n_airport'
    assert 'n_treatment' in data, 'variable not found in data: key=n_treatment'
    assert 'airport' in data, 'variable not found in data: key=airport'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_airport = data["n_airport"]
    n_treatment = data["n_treatment"]
    airport = data["airport"]
    treatment = data["treatment"]
    y = data["y"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(n_airport, low=0, dims=[1])
    check_constraints(n_treatment, low=0, dims=[1])
    check_constraints(airport, low=1, high=n_airport, dims=[N])
    check_constraints(treatment, low=1, high=n_treatment, dims=[N])
    check_constraints(y, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    n_airport = data["n_airport"]
    n_treatment = data["n_treatment"]
    airport = data["airport"]
    treatment = data["treatment"]
    y = data["y"]
    # assign init values for parameters
    params["d"] = init_vector("d", dims=(n_airport)) # vector
    params["g"] = init_vector("g", dims=(n_treatment)) # vector
    params["mu"] = init_real("mu") # real/double
    params["mu_d"] = init_real("mu_d") # real/double
    params["mu_g"] = init_real("mu_g") # real/double
    params["sigma_d"] = init_real("sigma_d", low=0, high=100) # real/double
    params["sigma_g"] = init_real("sigma_g", low=0, high=100) # real/double
    params["sigma_y"] = init_real("sigma_y", low=0, high=100) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n_airport = data["n_airport"]
    n_treatment = data["n_treatment"]
    airport = data["airport"]
    treatment = data["treatment"]
    y = data["y"]
    # INIT parameters
    d = params["d"]
    g = params["g"]
    mu = params["mu"]
    mu_d = params["mu_d"]
    mu_g = params["mu_g"]
    sigma_d = params["sigma_d"]
    sigma_g = params["sigma_g"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    d_adj = init_vector("d_adj", dims=(n_airport)) # vector
    g_adj = init_vector("g_adj", dims=(n_treatment)) # vector
    mu_adj = init_real("mu_adj") # real/double
    mu_d2 = init_real("mu_d2") # real/double
    mu_g2 = init_real("mu_g2") # real/double
    y_hat = init_vector("y_hat", dims=(N)) # vector
    mu_g2 = _pyro_assign(mu_g2, _call_func("mean", [g]))
    mu_d2 = _pyro_assign(mu_d2, _call_func("mean", [d]))
    g_adj = _pyro_assign(g_adj, _call_func("subtract", [g,mu_g2]))
    d_adj = _pyro_assign(d_adj, _call_func("subtract", [d,mu_d2]))
    mu_adj = _pyro_assign(mu_adj, ((mu + mu_g2) + mu_d2))
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], ((mu + _index_select(g, treatment[i - 1] - 1) ) + _index_select(d, airport[i - 1] - 1) ))
    # model block

    sigma_y =  _pyro_sample(sigma_y, "sigma_y", "uniform", [0, 100])
    sigma_d =  _pyro_sample(sigma_d, "sigma_d", "uniform", [0, 100])
    sigma_g =  _pyro_sample(sigma_g, "sigma_g", "uniform", [0, 100])
    mu =  _pyro_sample(mu, "mu", "normal", [0, 100])
    mu_g =  _pyro_sample(mu_g, "mu_g", "normal", [0, 1])
    mu_d =  _pyro_sample(mu_d, "mu_d", "normal", [0, 1])
    g =  _pyro_sample(g, "g", "normal", [(100 * mu_g), sigma_g])
    d =  _pyro_sample(d, "d", "normal", [(100 * mu_d), sigma_d])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

