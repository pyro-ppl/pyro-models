# model file: ../example-models/BPA/Ch.11/ipm_hoopoe.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'nyears' in data, 'variable not found in data: key=nyears'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'R' in data, 'variable not found in data: key=R'
    assert 'marray_j' in data, 'variable not found in data: key=marray_j'
    assert 'marray_a' in data, 'variable not found in data: key=marray_a'
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    marray_j = data["marray_j"]
    marray_a = data["marray_a"]
    check_constraints(nyears, dims=[1])
    check_constraints(y, dims=[nyears])
    check_constraints(J, dims=[(nyears - 1)])
    check_constraints(R, dims=[(nyears - 1)])
    check_constraints(marray_j, dims=[(nyears - 1), nyears])
    check_constraints(marray_a, dims=[(nyears - 1), nyears])

def transformed_data(data):
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    marray_j = data["marray_j"]
    marray_a = data["marray_a"]
    ny_minus_1 = init_int("ny_minus_1") # real/double
    data["ny_minus_1"] = ny_minus_1

def init_params(data, params):
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    marray_j = data["marray_j"]
    marray_a = data["marray_a"]
    # initialize transformed data
    ny_minus_1 = data["ny_minus_1"]
    # assign init values for parameters
    params["N1"] = init_vector("N1", low=0, dims=(nyears)) # vector
    params["NadSurv"] = init_vector("NadSurv", low=0, dims=(nyears)) # vector
    params["Nadimm"] = init_vector("Nadimm", low=0, dims=(nyears)) # vector
    params["l_mphij"] = init_real("l_mphij") # real/double
    params["l_mphia"] = init_real("l_mphia") # real/double
    params["l_mfec"] = init_real("l_mfec") # real/double
    params["l_mim"] = init_real("l_mim") # real/double
    params["l_p"] = init_real("l_p") # real/double
    params["epsilon_phij_raw"] = init_vector("epsilon_phij_raw", dims=(ny_minus_1)) # vector
    params["epsilon_phia_raw"] = init_vector("epsilon_phia_raw", dims=(ny_minus_1)) # vector
    params["epsilon_fec_raw"] = init_vector("epsilon_fec_raw", dims=(ny_minus_1)) # vector
    params["epsilon_im_raw"] = init_vector("epsilon_im_raw", dims=(ny_minus_1)) # vector
    params["sig_phij"] = init_real("sig_phij", low=0) # real/double
    params["sig_phia"] = init_real("sig_phia", low=0) # real/double
    params["sig_fec"] = init_real("sig_fec", low=0) # real/double
    params["sig_im"] = init_real("sig_im", low=0) # real/double

def model(data, params):
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    marray_j = data["marray_j"]
    marray_a = data["marray_a"]
    # initialize transformed data
    ny_minus_1 = data["ny_minus_1"]
    # INIT parameters
    N1 = params["N1"]
    NadSurv = params["NadSurv"]
    Nadimm = params["Nadimm"]
    l_mphij = params["l_mphij"]
    l_mphia = params["l_mphia"]
    l_mfec = params["l_mfec"]
    l_mim = params["l_mim"]
    l_p = params["l_p"]
    epsilon_phij_raw = params["epsilon_phij_raw"]
    epsilon_phia_raw = params["epsilon_phia_raw"]
    epsilon_fec_raw = params["epsilon_fec_raw"]
    epsilon_im_raw = params["epsilon_im_raw"]
    sig_phij = params["sig_phij"]
    sig_phia = params["sig_phia"]
    sig_fec = params["sig_fec"]
    sig_im = params["sig_im"]
    # initialize transformed parameters
    epsilon_phij = init_vector("epsilon_phij", dims=(ny_minus_1)) # vector
    epsilon_phia = init_vector("epsilon_phia", dims=(ny_minus_1)) # vector
    epsilon_fec = init_vector("epsilon_fec", dims=(ny_minus_1)) # vector
    epsilon_im = init_vector("epsilon_im", dims=(ny_minus_1)) # vector
    phij = init_vector("phij", low=0, high=1, dims=(ny_minus_1)) # vector
    phia = init_vector("phia", low=0, high=1, dims=(ny_minus_1)) # vector
    f = init_vector("f", low=0, dims=(ny_minus_1)) # vector
    omega = init_vector("omega", low=0, dims=(ny_minus_1)) # vector
    p = init_vector("p", low=0, high=1, dims=(ny_minus_1)) # vector
    Ntot = init_vector("Ntot", low=0, dims=(nyears)) # vector
    rho = init_vector("rho", low=0, dims=(ny_minus_1)) # vector
    pr_j = init_simplex("pr_j", dims=(ny_minus_1)) # real/double
    pr_a = init_simplex("pr_a", dims=(ny_minus_1)) # real/double
    epsilon_phij = _pyro_assign(epsilon_phij, _call_func("multiply", [sig_phij,epsilon_phij_raw]))
    epsilon_phia = _pyro_assign(epsilon_phia, _call_func("multiply", [sig_phia,epsilon_phia_raw]))
    epsilon_fec = _pyro_assign(epsilon_fec, _call_func("multiply", [sig_fec,epsilon_fec_raw]))
    epsilon_im = _pyro_assign(epsilon_im, _call_func("multiply", [sig_im,epsilon_im_raw]))
    for t in range(1, to_int((nyears - 1)) + 1):

        phij[t - 1] = _pyro_assign(phij[t - 1], _call_func("inv_logit", [(l_mphij + _index_select(epsilon_phij, t - 1) )]))
        phia[t - 1] = _pyro_assign(phia[t - 1], _call_func("inv_logit", [(l_mphia + _index_select(epsilon_phia, t - 1) )]))
        f[t - 1] = _pyro_assign(f[t - 1], _call_func("exp", [(l_mfec + _index_select(epsilon_fec, t - 1) )]))
        omega[t - 1] = _pyro_assign(omega[t - 1], _call_func("exp", [(l_mim + _index_select(epsilon_im, t - 1) )]))
        p[t - 1] = _pyro_assign(p[t - 1], _call_func("inv_logit", [l_p]))
    Ntot = _pyro_assign(Ntot, _call_func("add", [_call_func("add", [NadSurv,Nadimm]),N1]))
    pr_j = _pyro_assign(pr_j, _call_func("marray_juveniles", [nyears,phij,phia,p, pstream__]))
    pr_a = _pyro_assign(pr_a, _call_func("marray_adults", [nyears,phia,p, pstream__]))
    for t in range(1, to_int(ny_minus_1) + 1):
        rho[t - 1] = _pyro_assign(rho[t - 1], (_index_select(R, t - 1)  * _index_select(f, t - 1) ))
    # model block

    N1[1 - 1] =  _pyro_sample(_index_select(N1, 1 - 1) , "N1[%d]" % (to_int(1-1)), "normal", [100, 100])
    NadSurv[1 - 1] =  _pyro_sample(_index_select(NadSurv, 1 - 1) , "NadSurv[%d]" % (to_int(1-1)), "normal", [100, 100])
    Nadimm[1 - 1] =  _pyro_sample(_index_select(Nadimm, 1 - 1) , "Nadimm[%d]" % (to_int(1-1)), "normal", [100, 100])
    l_mphij =  _pyro_sample(l_mphij, "l_mphij", "normal", [0, 100])
    l_mphia =  _pyro_sample(l_mphia, "l_mphia", "normal", [0, 100])
    l_mfec =  _pyro_sample(l_mfec, "l_mfec", "normal", [0, 100])
    l_mim =  _pyro_sample(l_mim, "l_mim", "normal", [0, 100])
    l_p =  _pyro_sample(l_p, "l_p", "normal", [0, 100])
    epsilon_phij_raw =  _pyro_sample(epsilon_phij_raw, "epsilon_phij_raw", "normal", [0, 1])
    epsilon_phia_raw =  _pyro_sample(epsilon_phia_raw, "epsilon_phia_raw", "normal", [0, 1])
    epsilon_fec_raw =  _pyro_sample(epsilon_fec_raw, "epsilon_fec_raw", "normal", [0, 1])
    epsilon_im_raw =  _pyro_sample(epsilon_im_raw, "epsilon_im_raw", "normal", [0, 1])
    for t in range(2, to_int(nyears) + 1):
        # {
        mean1 = init_real("mean1") # real/double
        mpo = init_real("mpo") # real/double

        N1[t - 1] =  _pyro_sample(_index_select(N1, t - 1) , "N1[%d]" % (to_int(t-1)), "real_poisson", [mean1])
        NadSurv[t - 1] =  _pyro_sample(_index_select(NadSurv, t - 1) , "NadSurv[%d]" % (to_int(t-1)), "real_binomial", [_index_select(Ntot, (t - 1) - 1) , _index_select(phia, (t - 1) - 1) ])
        Nadimm[t - 1] =  _pyro_sample(_index_select(Nadimm, t - 1) , "Nadimm[%d]" % (to_int(t-1)), "real_poisson", [mpo])
        # }
    y =  _pyro_sample(y, "y", "poisson", [Ntot], obs=y)
    for t in range(1, to_int((nyears - 1)) + 1):

        marray_j[t - 1] =  _pyro_sample(_index_select(marray_j, t - 1) , "marray_j[%d]" % (to_int(t-1)), "multinomial", [_index_select(pr_j, t - 1) ], obs=_index_select(marray_j, t - 1) )
        marray_a[t - 1] =  _pyro_sample(_index_select(marray_a, t - 1) , "marray_a[%d]" % (to_int(t-1)), "multinomial", [_index_select(pr_a, t - 1) ], obs=_index_select(marray_a, t - 1) )
    J =  _pyro_sample(J, "J", "poisson", [rho], obs=J)

