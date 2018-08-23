# model file: ../example-models/BPA/Ch.06/Mb.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'M' in data, 'variable not found in data: key=M'
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    M = data["M"]
    T = data["T"]
    y = data["y"]
    check_constraints(M, low=0, dims=[1])
    check_constraints(T, low=0, dims=[1])
    check_constraints(y, low=0, high=1, dims=[M, T])

def transformed_data(data):
    # initialize data
    M = data["M"]
    T = data["T"]
    y = data["y"]
    s = init_int("s", low=0, dims=(M)) # real/double
    C = init_int("C", low=0) # real/double
    C = _pyro_assign(C, 0)
    for i in range(1, to_int(M) + 1):

        s[i - 1] = _pyro_assign(s[i - 1], _call_func("sum", [_index_select(y, i - 1) ]))
        if (as_bool(_call_func("logical_gt", [_index_select(s, i - 1) ,0]))):
            C = _pyro_assign(C, (C + 1))
        
    data["s"] = s
    data["C"] = C

def init_params(data, params):
    # initialize data
    M = data["M"]
    T = data["T"]
    y = data["y"]
    # initialize transformed data
    s = data["s"]
    C = data["C"]
    # assign init values for parameters
    params["omega"] = init_real("omega", low=0, high=1) # real/double
    params["p"] = init_real("p", low=0, high=1) # real/double
    params["c"] = init_real("c", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    M = data["M"]
    T = data["T"]
    y = data["y"]
    # initialize transformed data
    s = data["s"]
    C = data["C"]
    # INIT parameters
    omega = params["omega"]
    p = params["p"]
    c = params["c"]
    # initialize transformed parameters
    p_eff = init_vector("p_eff", low=0, high=1, dims=(M, T)) # vector
    for i in range(1, to_int(M) + 1):

        p_eff[i - 1][1 - 1] = _pyro_assign(p_eff[i - 1][1 - 1], p)
        for j in range(2, to_int(T) + 1):
            p_eff[i - 1][j - 1] = _pyro_assign(p_eff[i - 1][j - 1], (((1 - _index_select(_index_select(y, i - 1) , (j - 1) - 1) ) * p) + (_index_select(_index_select(y, i - 1) , (j - 1) - 1)  * c)))
    # model block

    for i in range(1, to_int(M) + 1):
        if (as_bool(_call_func("logical_gt", [_index_select(s, i - 1) ,0]))):
            pyro.sample("(_call_func( bernoulli_log , [1,omega]) + _call_func( bernoulli_log , [_index_select(y, i - 1) ,_index_select(p_eff, i - 1) ]))[%d]" % (i), dist.Bernoulli((_call_func("bernoulli_log", [1,omega]) + _call_func("bernoulli_log", [_index_select(y, i - 1) ,_index_select(p_eff, i - 1) ]))), obs=(1));
        else: 
            pyro.sample("_call_func( log_sum_exp , [(_call_func( bernoulli_log , [1,omega]) + _call_func( bernoulli_log , [0,_index_select(p_eff, i - 1) ])),_call_func( bernoulli_log , [0,omega])])[%d]" % (i), dist.Bernoulli(_call_func("log_sum_exp", [(_call_func("bernoulli_log", [1,omega]) + _call_func("bernoulli_log", [0,_index_select(p_eff, i - 1) ])),_call_func("bernoulli_log", [0,omega])])), obs=(1));
        

