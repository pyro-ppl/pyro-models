# model file: ../example-models/BPA/Ch.09/ms3_multinomlogit.stan
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
    check_constraints(y, low=1, high=4, dims=[nind, n_occasions])

def transformed_data(data):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    n_occ_minus_1 = init_int("n_occ_minus_1") # real/double
    first = init_int("first", low=0, high=n_occasions, dims=(nind)) # real/double
    for i in range(1, to_int(nind) + 1):
        first[i - 1] = _pyro_assign(first[i - 1], _call_func("first_capture", [_index_select(y, i - 1) , pstream__]))
    data["n_occ_minus_1"] = n_occ_minus_1
    data["first"] = first

def init_params(data, params):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    first = data["first"]
    # assign init values for parameters
    params["phiA"] = init_real("phiA", low=0, high=1) # real/double
    params["phiB"] = init_real("phiB", low=0, high=1) # real/double
    params["phiC"] = init_real("phiC", low=0, high=1) # real/double
    params["pA"] = init_real("pA", low=0, high=1) # real/double
    params["pB"] = init_real("pB", low=0, high=1) # real/double
    params["pC"] = init_real("pC", low=0, high=1) # real/double
    params["lpsiA"] = init_vector("lpsiA", dims=(2)) # vector
    params["lpsiB"] = init_vector("lpsiB", dims=(2)) # vector
    params["lpsiC"] = init_vector("lpsiC", dims=(2)) # vector

def model(data, params):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    first = data["first"]
    # INIT parameters
    phiA = params["phiA"]
    phiB = params["phiB"]
    phiC = params["phiC"]
    pA = params["pA"]
    pB = params["pB"]
    pC = params["pC"]
    lpsiA = params["lpsiA"]
    lpsiB = params["lpsiB"]
    lpsiC = params["lpsiC"]
    # initialize transformed parameters
    psiA = init_simplex("psiA") # real/double
    psiB = init_simplex("psiB") # real/double
    psiC = init_simplex("psiC") # real/double
    ps = init_simplex("ps", dims=(4, nind, n_occ_minus_1)) # real/double
    po = init_simplex("po", dims=(4, nind, n_occ_minus_1)) # real/double
    psiA = _pyro_assign(psiA, _call_func("softmax_0", [lpsiA, pstream__]))
    psiB = _pyro_assign(psiB, _call_func("softmax_0", [lpsiB, pstream__]))
    psiC = _pyro_assign(psiC, _call_func("softmax_0", [lpsiC, pstream__]))
    for i in range(1, to_int(nind) + 1):

        for t in range(1, to_int(n_occ_minus_1) + 1):

            ps[1 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][1 - 1], (phiA * _index_select(psiA, 1 - 1) ))
            ps[1 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][2 - 1], (phiA * _index_select(psiA, 2 - 1) ))
            ps[1 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][3 - 1], (phiA * _index_select(psiA, 3 - 1) ))
            ps[1 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][4 - 1], (1.0 - phiA))
            ps[2 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][1 - 1], (phiB * _index_select(psiB, 1 - 1) ))
            ps[2 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][2 - 1], (phiB * _index_select(psiB, 2 - 1) ))
            ps[2 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][3 - 1], (phiB * _index_select(psiB, 3 - 1) ))
            ps[2 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][4 - 1], (1.0 - phiB))
            ps[3 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][1 - 1], (phiC * _index_select(psiC, 1 - 1) ))
            ps[3 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][2 - 1], (phiC * _index_select(psiC, 2 - 1) ))
            ps[3 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][3 - 1], (phiC * _index_select(psiC, 3 - 1) ))
            ps[3 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][4 - 1], (1.0 - phiC))
            ps[4 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][1 - 1], 0.0)
            ps[4 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][2 - 1], 0.0)
            ps[4 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][3 - 1], 0.0)
            ps[4 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][4 - 1], 1.0)
            po[1 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][1 - 1], pA)
            po[1 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[1 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][3 - 1], 0.0)
            po[1 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][4 - 1], (1.0 - pA))
            po[2 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[2 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][2 - 1], pB)
            po[2 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][3 - 1], 0.0)
            po[2 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][4 - 1], (1.0 - pB))
            po[3 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[3 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[3 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][3 - 1], pC)
            po[3 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][4 - 1], (1.0 - pC))
            po[4 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[4 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[4 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][3 - 1], 0.0)
            po[4 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][4 - 1], 1.0)
    # model block
    # {
    acc = init_real("acc", dims=(4)) # real/double
    gamma = init_vector("gamma", dims=(n_occasions, 4)) # vector

    lpsiA =  _pyro_sample(lpsiA, "lpsiA", "normal", [0, _call_func("sqrt", [1000])])
    lpsiB =  _pyro_sample(lpsiB, "lpsiB", "normal", [0, _call_func("sqrt", [1000])])
    lpsiC =  _pyro_sample(lpsiC, "lpsiC", "normal", [0, _call_func("sqrt", [1000])])
    for i in range(1, to_int(nind) + 1):

        if (as_bool(_call_func("logical_gt", [_index_select(first, i - 1) ,0]))):

            for k in range(1, 4 + 1):
                gamma[first[i - 1] - 1][k - 1] = _pyro_assign(gamma[first[i - 1] - 1][k - 1], _call_func("logical_eq", [_index_select(_index_select(y, i - 1) , first[i - 1] - 1) ,k]))
            for t in range(to_int((first[i - 1] + 1)), to_int(n_occasions) + 1):

                for k in range(1, 4 + 1):

                    for j in range(1, 4 + 1):
                        acc[j - 1] = _pyro_assign(acc[j - 1], ((_index_select(_index_select(gamma, (t - 1) - 1) , j - 1)  * _index_select(_index_select(_index_select(_index_select(ps, j - 1) , i - 1) , (t - 1) - 1) , k - 1) ) * _index_select(_index_select(_index_select(_index_select(po, k - 1) , i - 1) , (t - 1) - 1) , y[i - 1][t - 1] - 1) ))
                    gamma[t - 1][k - 1] = _pyro_assign(gamma[t - 1][k - 1], _call_func("sum", [acc]))
            pyro.sample("_call_func( log , [_call_func( sum , [_index_select(gamma, n_occasions - 1) ])])[%d]" % (i), dist.Bernoulli(_call_func("log", [_call_func("sum", [_index_select(gamma, n_occasions - 1) ])])), obs=(1));
        
    # }

