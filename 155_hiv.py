# model file: ../example-models/ARM/Ch.20/hiv.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'person' in data, 'variable not found in data: key=person'
    assert 'time' in data, 'variable not found in data: key=time'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    y = data["y"]
    check_constraints(J, low=0, dims=[1])
    check_constraints(N, low=0, dims=[1])
    check_constraints(person, low=1, high=J, dims=[N])
    check_constraints(time, dims=[N])
    check_constraints(y, dims=[N])

def init_params(data, params):
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    y = data["y"]
    # assign init values for parameters
    params["a1"] = init_vector("a1", dims=(J)) # vector
    params["a2"] = init_vector("a2", dims=(J)) # vector
    params["mu_a1"] = init_real("mu_a1") # real/double
    params["mu_a2"] = init_real("mu_a2") # real/double
    params["sigma_a1"] = init_real("sigma_a1", low=0, high=100) # real/double
    params["sigma_a2"] = init_real("sigma_a2", low=0, high=100) # real/double
    params["sigma_y"] = init_real("sigma_y", low=0, high=100) # real/double

def model(data, params):
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    y = data["y"]
    # INIT parameters
    a1 = params["a1"]
    a2 = params["a2"]
    mu_a1 = params["mu_a1"]
    mu_a2 = params["mu_a2"]
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    y_hat = init_vector("y_hat", dims=(N)) # vector
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], (_index_select(a1, person[i - 1] - 1)  + (_index_select(a2, person[i - 1] - 1)  * _index_select(time, i - 1) )))
    # model block

    mu_a1 =  _pyro_sample(mu_a1, "mu_a1", "normal", [0, 1])
    mu_a2 =  _pyro_sample(mu_a2, "mu_a2", "normal", [0, 1])
    a1 =  _pyro_sample(a1, "a1", "normal", [mu_a1, sigma_a1])
    a2 =  _pyro_sample(a2, "a2", "normal", [(0.10000000000000001 * mu_a2), sigma_a2])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

