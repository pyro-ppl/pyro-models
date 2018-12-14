# model file: ../example-models/ARM/Ch.19/pilots_expansion.stan
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
    params["d_raw"] = init_vector("d_raw", dims=(n_airport)) # vector
    params["g_raw"] = init_vector("g_raw", dims=(n_treatment)) # vector
    params["mu"] = init_real("mu") # real/double
    params["mu_d_raw"] = init_real("mu_d_raw") # real/double
    params["mu_g_raw"] = init_real("mu_g_raw") # real/double
    params["sigma_d_raw"] = init_real("sigma_d_raw", low=0, high=100) # real/double
    params["sigma_g_raw"] = init_real("sigma_g_raw", low=0, high=100) # real/double
    params["sigma_y"] = init_real("sigma_y", low=0, high=100) # real/double
    params["xi_d"] = init_real("xi_d") # real/double
    params["xi_g"] = init_real("xi_g", low=0, high=100) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    n_airport = data["n_airport"]
    n_treatment = data["n_treatment"]
    airport = data["airport"]
    treatment = data["treatment"]
    y = data["y"]
    # INIT parameters
    d_raw = params["d_raw"]
    g_raw = params["g_raw"]
    mu = params["mu"]
    mu_d_raw = params["mu_d_raw"]
    mu_g_raw = params["mu_g_raw"]
    sigma_d_raw = params["sigma_d_raw"]
    sigma_g_raw = params["sigma_g_raw"]
    sigma_y = params["sigma_y"]
    xi_d = params["xi_d"]
    xi_g = params["xi_g"]
    # initialize transformed parameters
    d = init_vector("d", dims=(n_airport)) # vector
    g = init_vector("g", dims=(n_treatment)) # vector
    sigma_d = init_real("sigma_d", low=0) # real/double
    sigma_g = init_real("sigma_g", low=0) # real/double
    y_hat = init_vector("y_hat", dims=(N)) # vector
    g = _pyro_assign(g, _call_func("multiply", [xi_g,_call_func("subtract", [g_raw,_call_func("mean", [g_raw])])]))
    d = _pyro_assign(d, _call_func("multiply", [xi_d,_call_func("subtract", [d_raw,_call_func("mean", [d_raw])])]))
    sigma_g = _pyro_assign(sigma_g, (xi_g * sigma_g_raw))
    sigma_d = _pyro_assign(sigma_d, (_call_func("fabs", [xi_d]) * sigma_d_raw))
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], ((mu + _index_select(g, treatment[i - 1] - 1) ) + _index_select(d, airport[i - 1] - 1) ))
    # model block

    sigma_y =  _pyro_sample(sigma_y, "sigma_y", "uniform", [0, 100])
    sigma_d_raw =  _pyro_sample(sigma_d_raw, "sigma_d_raw", "uniform", [0, 100])
    sigma_g_raw =  _pyro_sample(sigma_g_raw, "sigma_g_raw", "uniform", [0, 100])
    xi_g =  _pyro_sample(xi_g, "xi_g", "uniform", [0, 100])
    xi_d =  _pyro_sample(xi_d, "xi_d", "normal", [0, 100])
    mu =  _pyro_sample(mu, "mu", "normal", [0, 100])
    mu_g_raw =  _pyro_sample(mu_g_raw, "mu_g_raw", "normal", [0, 1])
    mu_d_raw =  _pyro_sample(mu_d_raw, "mu_d_raw", "normal", [0, 1])
    g_raw =  _pyro_sample(g_raw, "g_raw", "normal", [(100 * mu_g_raw), sigma_g_raw])
    d_raw =  _pyro_sample(d_raw, "d_raw", "normal", [(100 * mu_d_raw), sigma_d_raw])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

