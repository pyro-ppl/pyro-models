# model file: ../example-models/BPA/Ch.08/mr_mnl_age3.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n_age' in data, 'variable not found in data: key=n_age'
    assert 'marr_j' in data, 'variable not found in data: key=marr_j'
    assert 'marr_a' in data, 'variable not found in data: key=marr_a'
    # initialize data
    n_age = data["n_age"]
    marr_j = data["marr_j"]
    marr_a = data["marr_a"]
    check_constraints(n_age, dims=[1])
    check_constraints(marr_j, dims=[(n_age + 1)])
    check_constraints(marr_a, dims=[(n_age + 1)])

def init_params(data, params):
    # initialize data
    n_age = data["n_age"]
    marr_j = data["marr_j"]
    marr_a = data["marr_a"]
    # assign init values for parameters
    params["sjuv"] = init_real("sjuv", low=0, high=1) # real/double
    params["ssub"] = init_real("ssub", low=0, high=1) # real/double
    params["sad"] = init_real("sad", low=0, high=1) # real/double
    params["rjuv"] = init_real("rjuv", low=0, high=1) # real/double
    params["rad"] = init_real("rad", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    n_age = data["n_age"]
    marr_j = data["marr_j"]
    marr_a = data["marr_a"]
    # INIT parameters
    sjuv = params["sjuv"]
    ssub = params["ssub"]
    sad = params["sad"]
    rjuv = params["rjuv"]
    rad = params["rad"]
    # initialize transformed parameters
    pr_a = init_simplex("pr_a") # real/double
    pr_j = init_simplex("pr_j") # real/double
    pr_j[1 - 1] = _pyro_assign(pr_j[1 - 1], ((1 - sjuv) * rjuv))
    pr_j[2 - 1] = _pyro_assign(pr_j[2 - 1], ((sjuv * (1 - ssub)) * rad))
    for t in range(3, to_int(n_age) + 1):
        pr_j[t - 1] = _pyro_assign(pr_j[t - 1], ((((sjuv * ssub) * _call_func("pow", [sad,(t - 3)])) * (1 - sad)) * rad))
    pr_j[(n_age + 1) - 1] = _pyro_assign(pr_j[(n_age + 1) - 1], (1 - _call_func("sum", [
