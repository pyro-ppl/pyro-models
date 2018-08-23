# model file: ../example-models/BPA/Ch.09/ms.stan
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
    check_constraints(y, low=1, high=3, dims=[nind, n_occasions])

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
    params["mean_phi"] = init_real("mean_phi", low=0, high=1, dims=(2)) # real/double
    params["mean_psi"] = init_real("mean_psi", low=0, high=1, dims=(2)) # real/double
    params["mean_p"] = init_real("mean_p", low=0, high=1, dims=(2)) # real/double

def model(data, params):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    first = data["first"]
    # INIT parameters
    mean_phi = params["mean_phi"]
    mean_psi = params["mean_psi"]
    mean_p = params["mean_p"]
    # initialize transformed parameters
    phiA = init_vector("phiA", low=0, high=1, dims=(n_occ_minus_1)) # vector
    phiB = init_vector("phiB", low=0, high=1, dims=(n_occ_minus_1)) # vector
    psiAB = init_vector("psiAB", low=0, high=1, dims=(n_occ_minus_1)) # vector
    psiBA = init_vector("psiBA", low=0, high=1, dims=(n_occ_minus_1)) # vector
    pA = init_vector("pA", low=0, high=1, dims=(n_occ_minus_1)) # vector
    pB = init_vector("pB", low=0, high=1, dims=(n_occ_minus_1)) # vector
    ps = init_simplex("ps", dims=(3, nind, n_occ_minus_1)) # real/double
    po = init_simplex("po", dims=(3, nind, n_occ_minus_1)) # real/double
    for t in range(1, to_int(n_occ_minus_1) + 1):

        phiA[t - 1] = _pyro_assign(phiA[t - 1], _index_select(mean_phi, 1 - 1) )
        phiB[t - 1] = _pyro_assign(phiB[t - 1], _index_select(mean_phi, 2 - 1) )
        psiAB[t - 1] = _pyro_assign(psiAB[t - 1], _index_select(mean_psi, 1 - 1) )
        psiBA[t - 1] = _pyro_assign(psiBA[t - 1], _index_select(mean_psi, 2 - 1) )
        pA[t - 1] = _pyro_assign(pA[t - 1], _index_select(mean_p, 1 - 1) )
        pB[t - 1] = _pyro_assign(pB[t - 1], _index_select(mean_p, 2 - 1) )
    for i in range(1, to_int(nind) + 1):

        for t in range(1, to_int((n_occasions - 1)) + 1):

            ps[1 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][1 - 1], (_index_select(phiA, t - 1)  * (1.0 - _index_select(psiAB, t - 1) )))
            ps[1 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][2 - 1], (_index_select(phiA, t - 1)  * _index_select(psiAB, t - 1) ))
            ps[1 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][3 - 1], (1.0 - _index_select(phiA, t - 1) ))
            ps[2 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][1 - 1], (_index_select(phiB, t - 1)  * _index_select(psiBA, t - 1) ))
            ps[2 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][2 - 1], (_index_select(phiB, t - 1)  * (1 - _index_select(psiBA, t - 1) )))
            ps[2 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][3 - 1], (1.0 - _index_select(phiB, t - 1) ))
            ps[3 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][1 - 1], 0.0)
            ps[3 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][2 - 1], 0.0)
            ps[3 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][3 - 1], 1.0)
            po[1 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][1 - 1], _index_select(pA, t - 1) )
            po[1 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[1 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][3 - 1], (1.0 - _index_select(pA, t - 1) ))
            po[2 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[2 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][2 - 1], _index_select(pB, t - 1) )
            po[2 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][3 - 1], (1.0 - _index_select(pB, t - 1) ))
            po[3 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[3 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[3 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][3 - 1], 1.0)
    # model block
    # {
    acc = init_real("acc", dims=(3)) # real/double
    gamma = init_vector("gamma", dims=(n_occasions, 3)) # vector

    for i in range(1, to_int(nind) + 1):

        if (as_bool(_call_func("logical_gt", [_index_select(first, i - 1) ,0]))):

            for k in range(1, 3 + 1):
                gamma[first[i - 1] - 1][k - 1] = _pyro_assign(gamma[first[i - 1] - 1][k - 1], _call_func("logical_eq", [k,_index_select(_index_select(y, i - 1) , first[i - 1] - 1) ]))
            for t in range(to_int((first[i - 1] + 1)), to_int(n_occasions) + 1):

                for k in range(1, 3 + 1):

                    for j in range(1, 3 + 1):
                        acc[j - 1] = _pyro_assign(acc[j - 1], ((_index_select(_index_select(gamma, (t - 1) - 1) , j - 1)  * _index_select(_index_select(_index_select(_index_select(ps, j - 1) , i - 1) , (t - 1) - 1) , k - 1) ) * _index_select(_index_select(_index_select(_index_select(po, k - 1) , i - 1) , (t - 1) - 1) , y[i - 1][t - 1] - 1) ))
                    gamma[t - 1][k - 1] = _pyro_assign(gamma[t - 1][k - 1], _call_func("sum", [acc]))
            pyro.sample("_call_func( log , [_call_func( sum , [_index_select(gamma, n_occasions - 1) ])])[%d]" % (i), dist.Bernoulli(_call_func("log", [_call_func("sum", [_index_select(gamma, n_occasions - 1) ])])), obs=(1));
        
    # }

