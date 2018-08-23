# model file: ../example-models/misc/ecology/mark-recapture/mark-recapture.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'M' in data, 'variable not found in data: key=M'
    assert 'C' in data, 'variable not found in data: key=C'
    assert 'R' in data, 'variable not found in data: key=R'
    # initialize data
    M = data["M"]
    C = data["C"]
    R = data["R"]
    check_constraints(M, low=0, dims=[1])
    check_constraints(C, low=0, dims=[1])
    check_constraints(R, low=0, high=_call_func("std::min", [M,C]), dims=[1])

def init_params(data, params):
    # initialize data
    M = data["M"]
    C = data["C"]
    R = data["R"]
    # assign init values for parameters
    params["N"] = init_real("N", low=((C - R) + M)) # real/double

def model(data, params):
    # initialize data
    M = data["M"]
    C = data["C"]
    R = data["R"]
    # INIT parameters
    N = params["N"]
    # initialize transformed parameters
    # model block

    R =  _pyro_sample(R, "R", "binomial", [C, (M / N)], obs=R)

