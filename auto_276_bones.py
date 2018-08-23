# model file: ../example-models/bugs_examples/vol1/bones/bones.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'nChild' in data, 'variable not found in data: key=nChild'
    assert 'nInd' in data, 'variable not found in data: key=nInd'
    assert 'gamma' in data, 'variable not found in data: key=gamma'
    assert 'delta' in data, 'variable not found in data: key=delta'
    assert 'ncat' in data, 'variable not found in data: key=ncat'
    assert 'grade' in data, 'variable not found in data: key=grade'
    # initialize data
    nChild = data["nChild"]
    nInd = data["nInd"]
    gamma = data["gamma"]
    delta = data["delta"]
    ncat = data["ncat"]
    grade = data["grade"]
    check_constraints(nChild, low=0, dims=[1])
    check_constraints(nInd, low=0, dims=[1])
    check_constraints(gamma, dims=[nInd, 4])
    check_constraints(delta, dims=[nInd])
    check_constraints(ncat, low=0, dims=[nInd])
    check_constraints(grade, dims=[nChild, nInd])

def init_params(data, params):
    # initialize data
    nChild = data["nChild"]
    nInd = data["nInd"]
    gamma = data["gamma"]
    delta = data["delta"]
    ncat = data["ncat"]
    grade = data["grade"]
    # assign init values for parameters
    params["theta"] = init_real("theta", dims=(nChild)) # real/double

def model(data, params):
    # initialize data
    nChild = data["nChild"]
    nInd = data["nInd"]
    gamma = data["gamma"]
    delta = data["delta"]
    ncat = data["ncat"]
    grade = data["grade"]
    # INIT parameters
    theta = params["theta"]
    # initialize transformed parameters
    # model block
    # {
    p = init_real("p", dims=(nChild, nInd, 5)) # real/double
    Q = init_real("Q", dims=(nChild, nInd, 4)) # real/double

    theta =  _pyro_sample(theta, "theta", "normal", [0.0, 36])
    for i in range(1, to_int(nChild) + 1):

        for j in range(1, to_int(nInd) + 1):

            for k in range(1, to_int((ncat[j - 1] - 1)) + 1):
                Q[i - 1][j - 1][k - 1] = _pyro_assign(Q[i - 1][j - 1][k - 1], _call_func("inv_logit", [(_index_select(delta, j - 1)  * (_index_select(theta, i - 1)  - _index_select(_index_select(gamma, j - 1) , k - 1) ))]))
            p[i - 1][j - 1][1 - 1] = _pyro_assign(p[i - 1][j - 1][1 - 1], (1 - _index_select(_index_select(_index_select(Q, i - 1) , j - 1) , 1 - 1) ))
            for k in range(2, to_int((ncat[j - 1] - 1)) + 1):
                p[i - 1][j - 1][k - 1] = _pyro_assign(p[i - 1][j - 1][k - 1], (_index_select(_index_select(_index_select(Q, i - 1) , j - 1) , (k - 1) - 1)  - _index_select(_index_select(_index_select(Q, i - 1) , j - 1) , k - 1) ))
            p[i - 1][j - 1][ncat[j - 1] - 1] = _pyro_assign(p[i - 1][j - 1][ncat[j - 1] - 1], _index_select(_index_select(_index_select(Q, i - 1) , j - 1) , (ncat[j - 1] - 1) - 1) )
            if (as_bool(_call_func("logical_neq", [_index_select(_index_select(grade, i - 1) , j - 1) ,-(1)]))):
                pyro.sample("_call_func( log , [_index_select(_index_select(_index_select(p, i - 1) , j - 1) , grade[i - 1][j - 1] - 1) ])[%d][%d]" % (i, j), dist.Bernoulli(_call_func("log", [_index_select(_index_select(_index_select(p, i - 1) , j - 1) , grade[i - 1][j - 1] - 1) ])), obs=(1));
            
    # }

