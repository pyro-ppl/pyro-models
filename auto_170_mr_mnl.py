# model file: ../example-models/BPA/Ch.08/mr_mnl.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n_occasions' in data, 'variable not found in data: key=n_occasions'
    assert 'marr' in data, 'variable not found in data: key=marr'
    # initialize data
    n_occasions = data["n_occasions"]
    marr = data["marr"]
    check_constraints(n_occasions, dims=[1])
    check_constraints(marr, dims=[n_occasions, (n_occasions + 1)])

def init_params(data, params):
    # initialize data
    n_occasions = data["n_occasions"]
    marr = data["marr"]
    # assign init values for parameters
    params["mean_s"] = init_real("mean_s", low=0, high=1) # real/double
    params["mean_r"] = init_real("mean_r", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    n_occasions = data["n_occasions"]
    marr = data["marr"]
    # INIT parameters
    mean_s = params["mean_s"]
    mean_r = params["mean_r"]
    # initialize transformed parameters
    s = init_vector("s", dims=(n_occasions)) # vector
    r = init_vector("r", dims=(n_occasions)) # vector
    pr = init_simplex("pr", dims=(n_occasions)) # real/double
    s = _pyro_assign(s, _call_func("rep_vector", [mean_s,n_occasions]))
    r = _pyro_assign(r, _call_func("rep_vector", [mean_r,n_occasions]))
    for t in range(1, to_int(n_occasions) + 1):

        pr[t - 1][t - 1] = _pyro_assign(pr[t - 1][t - 1], ((1 - _index_select(s, t - 1) ) * _index_select(r, t - 1) ))
        for j in range(to_int((t + 1)), to_int(n_occasions) + 1):
            pr[t - 1][j - 1] = _pyro_assign(pr[t - 1][j - 1], ((_call_func("prod", [
