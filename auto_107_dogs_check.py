# model file: ../example-models/ARM/Ch.24/dogs_check.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n_dogs' in data, 'variable not found in data: key=n_dogs'
    assert 'n_trials' in data, 'variable not found in data: key=n_trials'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    n_dogs = data["n_dogs"]
    n_trials = data["n_trials"]
    y = data["y"]
    check_constraints(n_dogs, low=0, dims=[1])
    check_constraints(n_trials, low=0, dims=[1])
    check_constraints(y, low=0, high=1, dims=[n_dogs, n_trials])

def init_params(data, params):
    # initialize data
    n_dogs = data["n_dogs"]
    n_trials = data["n_trials"]
    y = data["y"]
    # assign init values for parameters
    params["sigma_b1"] = init_real("sigma_b1", low=0, high=100) # real/double
    params["sigma_b2"] = init_real("sigma_b2", low=0, high=100) # real/double
    params["beta_neg"] = init_matrix("beta_neg", dims=(n_dogs, 2)) # matrix
    params["rho_b"] = init_real("rho_b", low=-(1), high=1) # real/double
    params["mu_beta"] = init_vector("mu_beta", dims=(2)) # vector

def model(data, params):
    # initialize data
    n_dogs = data["n_dogs"]
    n_trials = data["n_trials"]
    y = data["y"]
    # INIT parameters
    sigma_b1 = params["sigma_b1"]
    sigma_b2 = params["sigma_b2"]
    beta_neg = params["beta_neg"]
    rho_b = params["rho_b"]
    mu_beta = params["mu_beta"]
    # initialize transformed parameters
    # model block
    # {
    beta1 = init_vector("beta1", dims=(n_dogs)) # vector
    beta2 = init_vector("beta2", dims=(n_dogs)) # vector
    n_avoid = init_matrix("n_avoid", dims=(n_dogs, n_trials)) # matrix
    n_shock = init_matrix("n_shock", dims=(n_dogs, n_trials)) # matrix
    p = init_matrix("p", dims=(n_dogs, n_trials)) # matrix
    Sigma_b = init_matrix("Sigma_b", dims=(2, 2)) # matrix

    sigma_b1 =  _pyro_sample(sigma_b1, "sigma_b1", "uniform", [0, 100])
    sigma_b2 =  _pyro_sample(sigma_b2, "sigma_b2", "uniform", [0, 100])
    rho_b =  _pyro_sample(rho_b, "rho_b", "uniform", [-(1), 1])
    mu_beta =  _pyro_sample(mu_beta, "mu_beta", "normal", [0, 100])
    Sigma_b[1 - 1][1 - 1] = _pyro_assign(Sigma_b[1 - 1][1 - 1], _call_func("pow", [sigma_b1,2]))
    Sigma_b[2 - 1][2 - 1] = _pyro_assign(Sigma_b[2 - 1][2 - 1], _call_func("pow", [sigma_b2,2]))
    Sigma_b[1 - 1][2 - 1] = _pyro_assign(Sigma_b[1 - 1][2 - 1], ((rho_b * sigma_b1) * sigma_b2))
    Sigma_b[2 - 1][1 - 1] = _pyro_assign(Sigma_b[2 - 1][1 - 1], _index_select(_index_select(Sigma_b, 1 - 1) , 2 - 1) )
    for i in range(1, to_int(n_dogs) + 1):
        _call_func("transpose", [beta_neg[i - 1]]) =  _pyro_sample(_call_func("transpose", [_index_select(beta_neg, i - 1) ]), "_call_func( transpose , [beta_neg[i - 1]])", "multi_normal_prec", [mu_beta, Sigma_b])
    for j in range(1, to_int(n_dogs) + 1):

        n_avoid[j - 1][1 - 1] = _pyro_assign(n_avoid[j - 1][1 - 1], 0)
        n_shock[j - 1][1 - 1] = _pyro_assign(n_shock[j - 1][1 - 1], 0)
        beta1[j - 1] = _pyro_assign(beta1[j - 1], -(_call_func("exp", [_index_select(_index_select(beta_neg, j - 1) , 1 - 1) ])))
        beta2[j - 1] = _pyro_assign(beta2[j - 1], -(_call_func("exp", [_index_select(_index_select(beta_neg, j - 1) , 2 - 1) ])))
        for t in range(2, to_int(n_trials) + 1):

            n_avoid[j - 1][t - 1] = _pyro_assign(n_avoid[j - 1][t - 1], ((_index_select(_index_select(n_avoid, j - 1) , (t - 1) - 1)  + 1) - _index_select(_index_select(y, j - 1) , (t - 1) - 1) ))
            n_shock[j - 1][t - 1] = _pyro_assign(n_shock[j - 1][t - 1], (_index_select(_index_select(n_shock, j - 1) , (t - 1) - 1)  + _index_select(_index_select(y, j - 1) , (t - 1) - 1) ))
        for t in range(1, to_int(n_trials) + 1):

            p[j - 1][t - 1] = _pyro_assign(p[j - 1][t - 1], _call_func("inv_logit", [((_index_select(beta1, j - 1)  * _index_select(_index_select(n_avoid, j - 1) , t - 1) ) + (_index_select(beta2, j - 1)  * _index_select(_index_select(n_shock, j - 1) , t - 1) ))]))
            y[j - 1][t - 1] =  _pyro_sample(_index_select(_index_select(y, j - 1) , t - 1) , "y[%d][%d]" % (to_int(j-1),to_int(t-1)), "bernoulli", [_index_select(_index_select(p, j - 1) , t - 1) ], obs=_index_select(_index_select(y, j - 1) , t - 1) )
    # }

