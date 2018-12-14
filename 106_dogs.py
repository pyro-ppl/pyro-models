# model file: ../example-models/ARM/Ch.24/dogs.stan
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
    params["beta"] = init_vector("beta", dims=(3)) # vector

def model(data, params):
    # initialize data
    n_dogs = data["n_dogs"]
    n_trials = data["n_trials"]
    y = data["y"]
    # INIT parameters
    beta = params["beta"]
    # initialize transformed parameters
    n_avoid = init_matrix("n_avoid", dims=(n_dogs, n_trials)) # matrix
    n_shock = init_matrix("n_shock", dims=(n_dogs, n_trials)) # matrix
    p = init_matrix("p", dims=(n_dogs, n_trials)) # matrix
    for j in range(1, to_int(n_dogs) + 1):

        n_avoid[j - 1][1 - 1] = _pyro_assign(n_avoid[j - 1][1 - 1], 0)
        n_shock[j - 1][1 - 1] = _pyro_assign(n_shock[j - 1][1 - 1], 0)
        for t in range(2, to_int(n_trials) + 1):

            n_avoid[j - 1][t - 1] = _pyro_assign(n_avoid[j - 1][t - 1], ((_index_select(_index_select(n_avoid, j - 1) , (t - 1) - 1)  + 1) - _index_select(_index_select(y, j - 1) , (t - 1) - 1) ))
            n_shock[j - 1][t - 1] = _pyro_assign(n_shock[j - 1][t - 1], (_index_select(_index_select(n_shock, j - 1) , (t - 1) - 1)  + _index_select(_index_select(y, j - 1) , (t - 1) - 1) ))
        for t in range(1, to_int(n_trials) + 1):
            p[j - 1][t - 1] = _pyro_assign(p[j - 1][t - 1], ((_index_select(beta, 1 - 1)  + (_index_select(beta, 2 - 1)  * _index_select(_index_select(n_avoid, j - 1) , t - 1) )) + (_index_select(beta, 3 - 1)  * _index_select(_index_select(n_shock, j - 1) , t - 1) )))
    # model block

    beta =  _pyro_sample(beta, "beta", "normal", [0, 100])
    for i in range(1, to_int(n_dogs) + 1):

        for j in range(1, to_int(n_trials) + 1):
            y[i - 1][j - 1] =  _pyro_sample(_index_select(_index_select(y, i - 1) , j - 1) , "y[%d][%d]" % (to_int(i-1),to_int(j-1)), "bernoulli_logit", [_index_select(_index_select(p, i - 1) , j - 1) ], obs=_index_select(_index_select(y, i - 1) , j - 1) )

