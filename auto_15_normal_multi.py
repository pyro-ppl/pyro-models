# model file: ../example-models/basic_estimators/normal_multi.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'z' in data, 'variable not found in data: key=z'
    assert 'sigma' in data, 'variable not found in data: key=sigma'
    # initialize data
    N = data["N"]
    J = data["J"]
    y = data["y"]
    z = data["z"]
    sigma = data["sigma"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(J, low=0, dims=[1])
    check_constraints(y, dims=[J,N])
    check_constraints(z, dims=[J,N])
    check_constraints(sigma, dims=[N, N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    J = data["J"]
    y = data["y"]
    z = data["z"]
    sigma = data["sigma"]
    ry = 
