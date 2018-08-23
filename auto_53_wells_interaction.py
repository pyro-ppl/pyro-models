# model file: ../example-models/ARM/Ch.5/wells_interaction.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'arsenic' in data, 'variable not found in data: key=arsenic'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(switched, low=0, high=1, dims=[N])
    check_constraints(dist, dims=[N])
    check_constraints(arsenic, dims=[N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    dist100 = init_vector("dist100", dims=(N)) # vector
    inter = init_vector("inter", dims=(N)) # vector
    dist100 = _pyro_assign(dist100, _call_func("divide", [dist,100.0]))
    inter = _pyro_assign(inter, _call_func("elt_multiply", [dist100,arsenic]))
    data["dist100"] = dist100
    data["inter"] = inter

def init_params(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    # initialize transformed data
    dist100 = data["dist100"]
    inter = data["inter"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    # initialize transformed data
    dist100 = data["dist100"]
    inter = data["inter"]
    # INIT parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    switched =  _pyro_sample(switched, "switched", "bernoulli_logit", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,dist100])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,arsenic])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,inter])])], obs=switched)

