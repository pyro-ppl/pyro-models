# model file: ../example-models/BPA/Ch.03/GLM_Binomial.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'nyears' in data, 'variable not found in data: key=nyears'
    assert 'C' in data, 'variable not found in data: key=C'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'year' in data, 'variable not found in data: key=year'
    # initialize data
    nyears = data["nyears"]
    C = data["C"]
    N = data["N"]
    year = data["year"]
    check_constraints(nyears, low=0, dims=[1])
    check_constraints(C, low=0, dims=[nyears])
    check_constraints(N, low=0, dims=[nyears])
    check_constraints(year, dims=[nyears])

def transformed_data(data):
    # initialize data
    nyears = data["nyears"]
    C = data["C"]
    N = data["N"]
    year = data["year"]
    year_squared = init_vector("year_squared", dims=(nyears)) # vector
    year_squared = _pyro_assign(year_squared, _call_func("elt_multiply", [year,year]))
    data["year_squared"] = year_squared

def init_params(data, params):
    # initialize data
    nyears = data["nyears"]
    C = data["C"]
    N = data["N"]
    year = data["year"]
    # initialize transformed data
    year_squared = data["year_squared"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha") # real/double
    params["beta1"] = init_real("beta1") # real/double
    params["beta2"] = init_real("beta2") # real/double

def model(data, params):
    # initialize data
    nyears = data["nyears"]
    C = data["C"]
    N = data["N"]
    year = data["year"]
    # initialize transformed data
    year_squared = data["year_squared"]
    # INIT parameters
    alpha = params["alpha"]
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    # initialize transformed parameters
    logit_p = init_vector("logit_p", dims=(nyears)) # vector
    logit_p = _pyro_assign(logit_p, _call_func("add", [_call_func("add", [alpha,_call_func("multiply", [beta1,year])]),_call_func("multiply", [beta2,year_squared])]))
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0, 100])
    beta1 =  _pyro_sample(beta1, "beta1", "normal", [0, 100])
    beta2 =  _pyro_sample(beta2, "beta2", "normal", [0, 100])
    C =  _pyro_sample(C, "C", "binomial_logit", [N, logit_p], obs=C)

