# model file: ../example-models/BPA/Ch.08/mr_ss.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'nind' in data, 'variable not found in data: key=nind'
    assert 'n_occasions' in data, 'variable not found in data: key=n_occasions'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    check_constraints(nind, low=0, dims=[1])
    check_constraints(n_occasions, low=0, dims=[1])
    check_constraints(y, low=0, high=1, dims=[nind, n_occasions])

def transformed_data(data):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    n_occ_minus_1 = init_int("n_occ_minus_1") # real/double
    first = init_int("first", low=0, high=n_occasions, dims=(nind)) # real/double
    last = init_int("last", low=0, high=n_occasions, dims=(nind)) # real/double
    for i in range(1, to_int(nind) + 1):
        first[i - 1] = _pyro_assign(first[i - 1], _call_func("first_capture", [_index_select(y, i - 1) , pstream__]))
    for i in range(1, to_int(nind) + 1):
        last[i - 1] = _pyro_assign(last[i - 1], _call_func("last_capture", [_index_select(y, i - 1) , pstream__]))
    data["n_occ_minus_1"] = n_occ_minus_1
    data["first"] = first
    data["last"] = last

def init_params(data, params):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    first = data["first"]
    last = data["last"]
    # assign init values for parameters
    params["mean_s"] = init_real("mean_s", low=0, high=1) # real/double
    params["mean_r"] = init_real("mean_r", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    first = data["first"]
    last = data["last"]
    # INIT parameters
    mean_s = params["mean_s"]
    mean_r = params["mean_r"]
    # initialize transformed parameters
    s = init_matrix("s", low=0, high=1, dims=(nind, n_occ_minus_1)) # matrix
    r = init_matrix("r", low=0, high=1, dims=(nind, n_occ_minus_1)) # matrix
    for i in range(1, to_int(nind) + 1):

        for t in range(1, to_int((first[i - 1] - 1)) + 1):

            s[i - 1][t - 1] = _pyro_assign(s[i - 1][t - 1], 0)
            r[i - 1][t - 1] = _pyro_assign(r[i - 1][t - 1], 0)
        for t in range(to_int(first[i - 1]), to_int(n_occ_minus_1) + 1):

            s[i - 1][t - 1] = _pyro_assign(s[i - 1][t - 1], mean_s)
            r[i - 1][t - 1] = _pyro_assign(r[i - 1][t - 1], mean_r)
    # model block

    for i in range(1, to_int(nind) + 1):
        # {
        pr = init_real("pr") # real/double

        if (as_bool(_call_func("logical_gt", [_index_select(first, i - 1) ,0]))):

            if (as_bool(_call_func("logical_gt", [_index_select(last, i - 1) ,_index_select(first, i - 1) ]))):
                pr = _pyro_assign(pr, _call_func("cell_prob", [n_occasions,_index_select(s, i - 1) ,_index_select(r, i - 1) ,_index_select(first, i - 1) ,(_index_select(last, i - 1)  - 1), pstream__]))
            else: 
                pr = _pyro_assign(pr, _call_func("cell_prob", [n_occasions,_index_select(s, i - 1) ,_index_select(r, i - 1) ,_index_select(first, i - 1) ,n_occasions, pstream__]))
            
            
