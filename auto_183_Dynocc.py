# model file: ../example-models/BPA/Ch.13/Dynocc.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'nsite' in data, 'variable not found in data: key=nsite'
    assert 'nrep' in data, 'variable not found in data: key=nrep'
    assert 'nyear' in data, 'variable not found in data: key=nyear'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    nsite = data["nsite"]
    nrep = data["nrep"]
    nyear = data["nyear"]
    y = data["y"]
    check_constraints(nsite, low=1, dims=[1])
    check_constraints(nrep, low=1, dims=[1])
    check_constraints(nyear, low=1, dims=[1])
    check_constraints(y, low=0, high=1, dims=[nsite, nrep, nyear])

def transformed_data(data):
    # initialize data
    nsite = data["nsite"]
    nrep = data["nrep"]
    nyear = data["nyear"]
    y = data["y"]
    sum_y = init_int("sum_y", low=1, high=(nrep + 1), dims=(nsite, nyear)) # real/double
    ny_minus_1 = init_int("ny_minus_1") # real/double
    for i in range(1, to_int(nsite) + 1):
        for k in range(1, to_int(nyear) + 1):
            sum_y[i - 1][k - 1] = _pyro_assign(sum_y[i - 1][k - 1], (_call_func("sum", [
