# model file: ../example-models/BPA/Ch.09/agerecruitment.stan
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
    params["mean_phi1"] = init_real("mean_phi1", low=0, high=1) # real/double
    params["mean_phi2"] = init_real("mean_phi2", low=0, high=1) # real/double
    params["mean_phiad"] = init_real("mean_phiad", low=0, high=1) # real/double
    params["mean_alpha1"] = init_real("mean_alpha1", low=0, high=1) # real/double
    params["mean_alpha2"] = init_real("mean_alpha2", low=0, high=1) # real/double
    params["mean_pNB"] = init_real("mean_pNB", low=0, high=1) # real/double
    params["mean_pB"] = init_real("mean_pB", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    nind = data["nind"]
    n_occasions = data["n_occasions"]
    y = data["y"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    first = data["first"]
    # INIT parameters
    mean_phi1 = params["mean_phi1"]
    mean_phi2 = params["mean_phi2"]
    mean_phiad = params["mean_phiad"]
    mean_alpha1 = params["mean_alpha1"]
    mean_alpha2 = params["mean_alpha2"]
    mean_pNB = params["mean_pNB"]
    mean_pB = params["mean_pB"]
    # initialize transformed parameters
    phi_1 = init_vector("phi_1", low=0, high=1, dims=(n_occ_minus_1)) # vector
    phi_2 = init_vector("phi_2", low=0, high=1, dims=(n_occ_minus_1)) # vector
    phi_ad = init_vector("phi_ad", low=0, high=1, dims=(n_occ_minus_1)) # vector
    alpha_1 = init_vector("alpha_1", low=0, high=1, dims=(n_occ_minus_1)) # vector
    alpha_2 = init_vector("alpha_2", low=0, high=1, dims=(n_occ_minus_1)) # vector
    p_NB = init_vector("p_NB", low=0, high=1, dims=(n_occ_minus_1)) # vector
    p_B = init_vector("p_B", low=0, high=1, dims=(n_occ_minus_1)) # vector
    ps = init_simplex("ps", dims=(5, nind, n_occ_minus_1)) # real/double
    po = init_simplex("po", dims=(5, nind, n_occ_minus_1)) # real/double
    for t in range(1, to_int(n_occ_minus_1) + 1):

        phi_1[t - 1] = _pyro_assign(phi_1[t - 1], mean_phi1)
        phi_2[t - 1] = _pyro_assign(phi_2[t - 1], mean_phi2)
        phi_ad[t - 1] = _pyro_assign(phi_ad[t - 1], mean_phiad)
        alpha_1[t - 1] = _pyro_assign(alpha_1[t - 1], mean_alpha1)
        alpha_2[t - 1] = _pyro_assign(alpha_2[t - 1], mean_alpha2)
        p_NB[t - 1] = _pyro_assign(p_NB[t - 1], mean_pNB)
        p_B[t - 1] = _pyro_assign(p_B[t - 1], mean_pB)
    for i in range(1, to_int(nind) + 1):

        for t in range(1, to_int(n_occ_minus_1) + 1):

            ps[1 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][1 - 1], 0.0)
            ps[1 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][2 - 1], (_index_select(phi_1, t - 1)  * (1 - _index_select(alpha_1, t - 1) )))
            ps[1 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][3 - 1], 0.0)
            ps[1 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][4 - 1], (_index_select(phi_1, t - 1)  * _index_select(alpha_1, t - 1) ))
            ps[1 - 1][i - 1][t - 1][5 - 1] = _pyro_assign(ps[1 - 1][i - 1][t - 1][5 - 1], (1.0 - _index_select(phi_1, t - 1) ))
            ps[2 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][1 - 1], 0.0)
            ps[2 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][2 - 1], 0.0)
            ps[2 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][3 - 1], (_index_select(phi_2, t - 1)  * (1.0 - _index_select(alpha_2, t - 1) )))
            ps[2 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][4 - 1], (_index_select(phi_2, t - 1)  * _index_select(alpha_2, t - 1) ))
            ps[2 - 1][i - 1][t - 1][5 - 1] = _pyro_assign(ps[2 - 1][i - 1][t - 1][5 - 1], (1.0 - _index_select(phi_2, t - 1) ))
            ps[3 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][1 - 1], 0.0)
            ps[3 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][2 - 1], 0.0)
            ps[3 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][3 - 1], 0.0)
            ps[3 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][4 - 1], _index_select(phi_ad, t - 1) )
            ps[3 - 1][i - 1][t - 1][5 - 1] = _pyro_assign(ps[3 - 1][i - 1][t - 1][5 - 1], (1 - _index_select(phi_ad, t - 1) ))
            ps[4 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][1 - 1], 0.0)
            ps[4 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][2 - 1], 0.0)
            ps[4 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][3 - 1], 0.0)
            ps[4 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][4 - 1], _index_select(phi_ad, t - 1) )
            ps[4 - 1][i - 1][t - 1][5 - 1] = _pyro_assign(ps[4 - 1][i - 1][t - 1][5 - 1], (1.0 - _index_select(phi_ad, t - 1) ))
            ps[5 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(ps[5 - 1][i - 1][t - 1][1 - 1], 0.0)
            ps[5 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(ps[5 - 1][i - 1][t - 1][2 - 1], 0.0)
            ps[5 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(ps[5 - 1][i - 1][t - 1][3 - 1], 0.0)
            ps[5 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(ps[5 - 1][i - 1][t - 1][4 - 1], 0.0)
            ps[5 - 1][i - 1][t - 1][5 - 1] = _pyro_assign(ps[5 - 1][i - 1][t - 1][5 - 1], 1.0)
            po[1 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[1 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[1 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][3 - 1], 0.0)
            po[1 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[1 - 1][i - 1][t - 1][4 - 1], 1.0)
            po[2 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[2 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][2 - 1], _index_select(p_NB, t - 1) )
            po[2 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][3 - 1], 0.0)
            po[2 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[2 - 1][i - 1][t - 1][4 - 1], (1.0 - _index_select(p_NB, t - 1) ))
            po[3 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[3 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][2 - 1], _index_select(p_NB, t - 1) )
            po[3 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][3 - 1], 0.0)
            po[3 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[3 - 1][i - 1][t - 1][4 - 1], (1.0 - _index_select(p_NB, t - 1) ))
            po[4 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[4 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[4 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][3 - 1], _index_select(p_B, t - 1) )
            po[4 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[4 - 1][i - 1][t - 1][4 - 1], (1.0 - _index_select(p_B, t - 1) ))
            po[5 - 1][i - 1][t - 1][1 - 1] = _pyro_assign(po[5 - 1][i - 1][t - 1][1 - 1], 0.0)
            po[5 - 1][i - 1][t - 1][2 - 1] = _pyro_assign(po[5 - 1][i - 1][t - 1][2 - 1], 0.0)
            po[5 - 1][i - 1][t - 1][3 - 1] = _pyro_assign(po[5 - 1][i - 1][t - 1][3 - 1], 0.0)
            po[5 - 1][i - 1][t - 1][4 - 1] = _pyro_assign(po[5 - 1][i - 1][t - 1][4 - 1], 1.0)
    # model block
    # {
    acc = init_real("acc", dims=(5)) # real/double
    gamma = init_vector("gamma", dims=(n_occasions, 5)) # vector

    for i in range(1, to_int(nind) + 1):

        if (as_bool(_call_func("logical_gt", [_index_select(first, i - 1) ,0]))):

            for k in range(1, 5 + 1):
                gamma[first[i - 1] - 1][k - 1] = _pyro_assign(gamma[first[i - 1] - 1][k - 1], _call_func("logical_eq", [k,_index_select(_index_select(y, i - 1) , first[i - 1] - 1) ]))
            for t in range(to_int((first[i - 1] + 1)), to_int(n_occasions) + 1):

                for k in range(1, 5 + 1):

                    for j in range(1, 5 + 1):
                        acc[j - 1] = _pyro_assign(acc[j - 1], ((_index_select(_index_select(gamma, (t - 1) - 1) , j - 1)  * _index_select(_index_select(_index_select(_index_select(ps, j - 1) , i - 1) , (t - 1) - 1) , k - 1) ) * _index_select(_index_select(_index_select(_index_select(po, k - 1) , i - 1) , (t - 1) - 1) , y[i - 1][t - 1] - 1) ))
                    gamma[t - 1][k - 1] = _pyro_assign(gamma[t - 1][k - 1], _call_func("sum", [acc]))
            pyro.sample("_call_func( log , [_call_func( sum , [_index_select(gamma, n_occasions - 1) ])])[%d]" % (i), dist.Bernoulli(_call_func("log", [_call_func("sum", [_index_select(gamma, n_occasions - 1) ])])), obs=(1));
        
    # }

