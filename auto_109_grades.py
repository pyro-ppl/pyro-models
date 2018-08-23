# model file: ../example-models/ARM/Ch.8/grades.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'final' in data, 'variable not found in data: key=final'
    assert 'midterm' in data, 'variable not found in data: key=midterm'
    # initialize data
    N = data["N"]
    final = data["final"]
    midterm = data["midterm"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(final, dims=[N])
    check_constraints(midterm, dims=[N])

def init_params(data, params):
    # initialize data
    N = data["N"]
    final = data["final"]
    midterm = data["midterm"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(2)) # vector
    params["sigma"] = init_real("sigma", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    final = data["final"]
    midterm = data["midterm"]
    # INIT parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    final =  _pyro_sample(final, "final", "normal", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,midterm])]), sigma], obs=final)

