# model file: ../example-models/BPA/Ch.03/GLM_Poisson.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'n' in data, 'variable not found in data: key=n'
    assert 'C' in data, 'variable not found in data: key=C'
    assert 'year' in data, 'variable not found in data: key=year'
    # initialize data
    n = data["n"]
    C = data["C"]
    year = data["year"]
    check_constraints(n, low=0, dims=[1])
    check_constraints(C, low=0, dims=[n])
    check_constraints(year, dims=[n])

def transformed_data(data):
    # initialize data
    n = data["n"]
    C = data["C"]
    year = data["year"]
    year_squared = init_vector("year_squared", dims=(n)) # vector
    year_cubed = init_vector("year_cubed", dims=(n)) # vector
    year_squared = _pyro_assign(year_squared, _call_func("elt_multiply", [year,year]))
    year_cubed = _pyro_assign(year_cubed, _call_func("elt_multiply", [_call_func("elt_multiply", [year,year]),year]))
    data["year_squared"] = year_squared
    data["year_cubed"] = year_cubed

def init_params(data, params):
    # initialize data
    n = data["n"]
    C = data["C"]
    year = data["year"]
    # initialize transformed data
    year_squared = data["year_squared"]
    year_cubed = data["year_cubed"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha", low=-(20), high=20) # real/double
    params["beta1"] = init_real("beta1", low=-(10), high=10) # real/double
    params["beta2"] = init_real("beta2", low=-(10), high=10) # real/double
    params["beta3"] = init_real("beta3", low=-(10), high=10) # real/double

def model(data, params):
    # initialize data
    n = data["n"]
    C = data["C"]
    year = data["year"]
    # initialize transformed data
    year_squared = data["year_squared"]
    year_cubed = data["year_cubed"]
    # INIT parameters
    alpha = params["alpha"]
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    beta3 = params["beta3"]
    # initialize transformed parameters
    log_lambda = init_vector("log_lambda", dims=(n)) # vector
    log_lambda = _pyro_assign(log_lambda, _call_func("add", [_call_func("add", [_call_func("add", [alpha,_call_func("multiply", [beta1,year])]),_call_func("multiply", [beta2,year_squared])]),_call_func("multiply", [beta3,year_cubed])]))
    # model block

    C =  _pyro_sample(C, "C", "poisson_log", [log_lambda], obs=C)

