# model file: ../example-models/BPA/Ch.07/cjs_mnl_age.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n_occasions' in data, 'variable not found in data: key=n_occasions'
    assert 'marr_j' in data, 'variable not found in data: key=marr_j'
    assert 'marr_a' in data, 'variable not found in data: key=marr_a'
    # initialize data
    n_occasions = data["n_occasions"]
    marr_j = data["marr_j"]
    marr_a = data["marr_a"]
    check_constraints(n_occasions, low=0, dims=[1])
    check_constraints(marr_j, low=0, dims=[(n_occasions - 1), n_occasions])
    check_constraints(marr_a, low=0, dims=[(n_occasions - 1), n_occasions])

def transformed_data(data):
    # initialize data
    n_occasions = data["n_occasions"]
    marr_j = data["marr_j"]
    marr_a = data["marr_a"]
    n_occ_minus_1 = init_int("n_occ_minus_1") # real/double
    data["n_occ_minus_1"] = n_occ_minus_1

def init_params(data, params):
    # initialize data
    n_occasions = data["n_occasions"]
    marr_j = data["marr_j"]
    marr_a = data["marr_a"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    # assign init values for parameters
    params["mean_phijuv"] = init_real("mean_phijuv", low=0, high=1) # real/double
    params["mean_phiad"] = init_real("mean_phiad", low=0, high=1) # real/double
    params["mean_p"] = init_real("mean_p", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    n_occasions = data["n_occasions"]
    marr_j = data["marr_j"]
    marr_a = data["marr_a"]
    # initialize transformed data
    n_occ_minus_1 = data["n_occ_minus_1"]
    # INIT parameters
    mean_phijuv = params["mean_phijuv"]
    mean_phiad = params["mean_phiad"]
    mean_p = params["mean_p"]
    # initialize transformed parameters
    phi_juv = init_vector("phi_juv", low=0, high=1, dims=(n_occ_minus_1)) # vector
    phi_ad = init_vector("phi_ad", low=0, high=1, dims=(n_occ_minus_1)) # vector
    p = init_vector("p", low=0, high=1, dims=(n_occ_minus_1)) # vector
    q = init_vector("q", low=0, high=1, dims=(n_occ_minus_1)) # vector
    pr_j = init_simplex("pr_j", dims=(n_occ_minus_1)) # real/double
    pr_a = init_simplex("pr_a", dims=(n_occ_minus_1)) # real/double
    phi_juv = _pyro_assign(phi_juv, _call_func("rep_vector", [mean_phijuv,n_occ_minus_1]))
    phi_ad = _pyro_assign(phi_ad, _call_func("rep_vector", [mean_phiad,n_occ_minus_1]))
    p = _pyro_assign(p, _call_func("rep_vector", [mean_p,n_occ_minus_1]))
    q = _pyro_assign(q, _call_func("subtract", [1.0,p]))
    for t in range(1, to_int(n_occ_minus_1) + 1):

        pr_j[t - 1][t - 1] = _pyro_assign(pr_j[t - 1][t - 1], (_index_select(phi_juv, t - 1)  * _index_select(p, t - 1) ))
        pr_a[t - 1][t - 1] = _pyro_assign(pr_a[t - 1][t - 1], (_index_select(phi_ad, t - 1)  * _index_select(p, t - 1) ))
        for j in range(to_int((t + 1)), to_int(n_occ_minus_1) + 1):

            pr_j[t - 1][j - 1] = _pyro_assign(pr_j[t - 1][j - 1], (((_index_select(phi_juv, t - 1)  * _call_func("prod", [
