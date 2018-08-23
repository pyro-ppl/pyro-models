# model file: ../example-models/BPA/Ch.13/site-occ.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'R' in data, 'variable not found in data: key=R'
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    check_constraints(R, low=1, dims=[1])
    check_constraints(T, low=1, dims=[1])
    check_constraints(y, low=0, high=1, dims=[R, T])

def transformed_data(data):
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    sum_y = init_int("sum_y", low=0, high=T, dims=(R)) # real/double
    occ_obs = init_int("occ_obs", low=0, high=R) # real/double
    occ_obs = _pyro_assign(occ_obs, 0)
    for i in range(1, to_int(R) + 1):

        sum_y[i - 1] = _pyro_assign(sum_y[i - 1], _call_func("sum", [_index_select(y, i - 1) ]))
        if (as_bool(_index_select(sum_y, i - 1) )):
            occ_obs = _pyro_assign(occ_obs, (occ_obs + 1))
        
    data["sum_y"] = sum_y
    data["occ_obs"] = occ_obs

def init_params(data, params):
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    # initialize transformed data
    sum_y = data["sum_y"]
    occ_obs = data["occ_obs"]
    # assign init values for parameters
    params["psi"] = init_real("psi", low=0, high=1) # real/double
    params["p"] = init_real("p", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    R = data["R"]
    T = data["T"]
    y = data["y"]
    # initialize transformed data
    sum_y = data["sum_y"]
    occ_obs = data["occ_obs"]
    # INIT parameters
    psi = params["psi"]
    p = params["p"]
    # initialize transformed parameters
    # model block

    for i in range(1, to_int(R) + 1):

        if (as_bool(_index_select(sum_y, i - 1) )):

            
