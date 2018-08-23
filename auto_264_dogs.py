# model file: ../example-models/bugs_examples/vol1/dogs/dogs.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'Ndogs' in data, 'variable not found in data: key=Ndogs'
    assert 'Ntrials' in data, 'variable not found in data: key=Ntrials'
    assert 'Y' in data, 'variable not found in data: key=Y'
    # initialize data
    Ndogs = data["Ndogs"]
    Ntrials = data["Ntrials"]
    Y = data["Y"]
    check_constraints(Ndogs, low=0, dims=[1])
    check_constraints(Ntrials, low=0, dims=[1])
    check_constraints(Y, dims=[Ndogs, Ntrials])

def transformed_data(data):
    # initialize data
    Ndogs = data["Ndogs"]
    Ntrials = data["Ntrials"]
    Y = data["Y"]
    y = init_int("y", dims=(Ndogs, Ntrials)) # real/double
    xa = init_int("xa", dims=(Ndogs, Ntrials)) # real/double
    xs = init_int("xs", dims=(Ndogs, Ntrials)) # real/double
    for dog in range(1, to_int(Ndogs) + 1):

        xa[dog - 1][1 - 1] = _pyro_assign(xa[dog - 1][1 - 1], 0)
        xs[dog - 1][1 - 1] = _pyro_assign(xs[dog - 1][1 - 1], 0)
        for trial in range(2, to_int(Ntrials) + 1):

            for k in range(1, to_int((trial - 1)) + 1):
                xa[dog - 1][trial - 1] = _pyro_assign(xa[dog - 1][trial - 1], (_index_select(_index_select(xa, dog - 1) , trial - 1)  + _index_select(_index_select(Y, dog - 1) , k - 1) ))
            xs[dog - 1][trial - 1] = _pyro_assign(xs[dog - 1][trial - 1], ((trial - 1) - _index_select(_index_select(xa, dog - 1) , trial - 1) ))
    for dog in range(1, to_int(Ndogs) + 1):

        for trial in range(1, to_int(Ntrials) + 1):

            y[dog - 1][trial - 1] = _pyro_assign(y[dog - 1][trial - 1], (1 - _index_select(_index_select(Y, dog - 1) , trial - 1) ))
    data["y"] = y
    data["xa"] = xa
    data["xs"] = xs

def init_params(data, params):
    # initialize data
    Ndogs = data["Ndogs"]
    Ntrials = data["Ntrials"]
    Y = data["Y"]
    # initialize transformed data
    y = data["y"]
    xa = data["xa"]
    xs = data["xs"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha", high=-(1.0000000000000001e-05)) # real/double
    params["beta"] = init_real("beta", high=-(1.0000000000000001e-05)) # real/double

def model(data, params):
    # initialize data
    Ndogs = data["Ndogs"]
    Ntrials = data["Ntrials"]
    Y = data["Y"]
    # initialize transformed data
    y = data["y"]
    xa = data["xa"]
    xs = data["xs"]
    # INIT parameters
    alpha = params["alpha"]
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0.0, 316.19999999999999])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0, 316.19999999999999])
    for dog in range(1, to_int(Ndogs) + 1):
        for trial in range(2, to_int(Ntrials) + 1):
            y[dog - 1][trial - 1] =  _pyro_sample(_index_select(_index_select(y, dog - 1) , trial - 1) , "y[%d][%d]" % (to_int(dog-1),to_int(trial-1)), "bernoulli", [_call_func("exp", [((alpha * _index_select(_index_select(xa, dog - 1) , trial - 1) ) + (beta * _index_select(_index_select(xs, dog - 1) , trial - 1) ))])], obs=_index_select(_index_select(y, dog - 1) , trial - 1) )

