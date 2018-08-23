# model file: ../example-models/bugs_examples/vol2/ice/ice.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'Nage' in data, 'variable not found in data: key=Nage'
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'year' in data, 'variable not found in data: key=year'
    assert 'cases' in data, 'variable not found in data: key=cases'
    assert 'age' in data, 'variable not found in data: key=age'
    assert 'pyr' in data, 'variable not found in data: key=pyr'
    assert 'alpha1' in data, 'variable not found in data: key=alpha1'
    # initialize data
    N = data["N"]
    Nage = data["Nage"]
    K = data["K"]
    year = data["year"]
    cases = data["cases"]
    age = data["age"]
    pyr = data["pyr"]
    alpha1 = data["alpha1"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(Nage, low=0, dims=[1])
    check_constraints(K, low=0, dims=[1])
    check_constraints(year, dims=[N])
    check_constraints(cases, dims=[N])
    check_constraints(age, dims=[N])
    check_constraints(pyr, dims=[N])
    check_constraints(alpha1, dims=[1])

def init_params(data, params):
    # initialize data
    N = data["N"]
    Nage = data["Nage"]
    K = data["K"]
    year = data["year"]
    cases = data["cases"]
    age = data["age"]
    pyr = data["pyr"]
    alpha1 = data["alpha1"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha", dims=((Nage - 1))) # real/double
    params["beta"] = init_real("beta", dims=(K)) # real/double
    params["sigma"] = init_real("sigma", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    Nage = data["Nage"]
    K = data["K"]
    year = data["year"]
    cases = data["cases"]
    age = data["age"]
    pyr = data["pyr"]
    alpha1 = data["alpha1"]
    # INIT parameters
    alpha = params["alpha"]
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block
    # {
    r = init_vector("r", dims=(N)) # vector

    sigma =  _pyro_sample(sigma, "sigma", "uniform", [0, 1])
    for k in range(1, 2 + 1):
        beta[k - 1] =  _pyro_sample(_index_select(beta, k - 1) , "beta[%d]" % (to_int(k-1)), "normal", [0, (sigma * 1000.0)])
    for k in range(3, to_int(K) + 1):
        beta[k - 1] =  _pyro_sample(_index_select(beta, k - 1) , "beta[%d]" % (to_int(k-1)), "normal", [((2 * _index_select(beta, (k - 1) - 1) ) - _index_select(beta, (k - 2) - 1) ), sigma])
    alpha =  _pyro_sample(alpha, "alpha", "normal", [0, 1000])
    for i in range(1, to_int(N) + 1):

        if (as_bool(_call_func("logical_eq", [_index_select(age, i - 1) ,1]))):
            r[i - 1] = _pyro_assign(r[i - 1], ((alpha1 + _call_func("log", [_index_select(pyr, i - 1) ])) + _index_select(beta, year[i - 1] - 1) ))
        else: 
            r[i - 1] = _pyro_assign(r[i - 1], ((_index_select(alpha, (age[i - 1] - 1) - 1)  + _call_func("log", [_index_select(pyr, i - 1) ])) + _index_select(beta, year[i - 1] - 1) ))
        
    cases =  _pyro_sample(cases, "cases", "poisson_log", [r], obs=cases)
    # }

