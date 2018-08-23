# model file: ../example-models/bugs_examples/vol2/endo/endo2.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n10' in data, 'variable not found in data: key=n10'
    assert 'n01' in data, 'variable not found in data: key=n01'
    assert 'n11' in data, 'variable not found in data: key=n11'
    assert 'I' in data, 'variable not found in data: key=I'
    assert 'J' in data, 'variable not found in data: key=J'
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    J = data["J"]
    check_constraints(n10, dims=[1])
    check_constraints(n01, dims=[1])
    check_constraints(n11, dims=[1])
    check_constraints(I, dims=[1])
    check_constraints(J, dims=[1])

def transformed_data(data):
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    J = data["J"]
    Y = init_int("Y", low=0, dims=(I, 2)) # real/double
    est = init_int("est", low=0, dims=(I, 2)) # real/double
    for i in range(1, to_int(I) + 1):

        Y[i - 1][1 - 1] = _pyro_assign(Y[i - 1][1 - 1], 1)
        Y[i - 1][2 - 1] = _pyro_assign(Y[i - 1][2 - 1], 0)
    for i in range(1, to_int(n10) + 1):

        est[i - 1][1 - 1] = _pyro_assign(est[i - 1][1 - 1], 1)
        est[i - 1][2 - 1] = _pyro_assign(est[i - 1][2 - 1], 0)
    for i in range(to_int((n10 + 1)), to_int((n10 + n01)) + 1):

        est[i - 1][1 - 1] = _pyro_assign(est[i - 1][1 - 1], 0)
        est[i - 1][2 - 1] = _pyro_assign(est[i - 1][2 - 1], 1)
    for i in range(to_int(((n10 + n01) + 1)), to_int(((n10 + n01) + n11)) + 1):

        est[i - 1][1 - 1] = _pyro_assign(est[i - 1][1 - 1], 1)
        est[i - 1][2 - 1] = _pyro_assign(est[i - 1][2 - 1], 1)
    for i in range(to_int((((n10 + n01) + n11) + 1)), to_int(I) + 1):

        est[i - 1][1 - 1] = _pyro_assign(est[i - 1][1 - 1], 0)
        est[i - 1][2 - 1] = _pyro_assign(est[i - 1][2 - 1], 0)
    data["Y"] = Y
    data["est"] = est

def init_params(data, params):
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    J = data["J"]
    # initialize transformed data
    Y = data["Y"]
    est = data["est"]
    # assign init values for parameters
    params["beta"] = init_real("beta") # real/double

def model(data, params):
    # initialize data
    n10 = data["n10"]
    n01 = data["n01"]
    n11 = data["n11"]
    I = data["I"]
    J = data["J"]
    # initialize transformed data
    Y = data["Y"]
    est = data["est"]
    # INIT parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block
    # {
    p = init_real("p", dims=(I, 2)) # real/double

    beta =  _pyro_sample(beta, "beta", "normal", [0, 1000])
    for i in range(1, to_int(I) + 1):

        p[i - 1][1 - 1] = _pyro_assign(p[i - 1][1 - 1], _call_func("exp", [(beta * _index_select(_index_select(est, i - 1) , 1 - 1) )]))
        p[i - 1][2 - 1] = _pyro_assign(p[i - 1][2 - 1], _call_func("exp", [(beta * _index_select(_index_select(est, i - 1) , 2 - 1) )]))
        p[i - 1][1 - 1] = _pyro_assign(p[i - 1][1 - 1], (_index_select(_index_select(p, i - 1) , 1 - 1)  / (_index_select(_index_select(p, i - 1) , 1 - 1)  + _index_select(_index_select(p, i - 1) , 2 - 1) )))
        p[i - 1][2 - 1] = _pyro_assign(p[i - 1][2 - 1], (1 - _index_select(_index_select(p, i - 1) , 1 - 1) ))
        pyro.sample("(_call_func( log , [_index_select(_index_select(p, i - 1) , 1 - 1) ]) * _index_select(_index_select(Y, i - 1) , 1 - 1) )[%d]" % (i), dist.Bernoulli((_call_func("log", [_index_select(_index_select(p, i - 1) , 1 - 1) ]) * _index_select(_index_select(Y, i - 1) , 1 - 1) )), obs=(1));
        pyro.sample("(_call_func( log , [_index_select(_index_select(p, i - 1) , 2 - 1) ]) * _index_select(_index_select(Y, i - 1) , 2 - 1) )[%d]" % (i), dist.Bernoulli((_call_func("log", [_index_select(_index_select(p, i - 1) , 2 - 1) ]) * _index_select(_index_select(Y, i - 1) , 2 - 1) )), obs=(1));
    # }

