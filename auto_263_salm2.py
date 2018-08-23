# model file: ../example-models/bugs_examples/vol1/salm/salm2.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'Ndoses' in data, 'variable not found in data: key=Ndoses'
    assert 'Nplates' in data, 'variable not found in data: key=Nplates'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'x' in data, 'variable not found in data: key=x'
    # initialize data
    Ndoses = data["Ndoses"]
    Nplates = data["Nplates"]
    y = data["y"]
    x = data["x"]
    check_constraints(Ndoses, low=0, dims=[1])
    check_constraints(Nplates, low=0, dims=[1])
    check_constraints(y, low=0, dims=[Ndoses, Nplates])
    check_constraints(x, low=0, dims=[Ndoses])

def init_params(data, params):
    # initialize data
    Ndoses = data["Ndoses"]
    Nplates = data["Nplates"]
    y = data["y"]
    x = data["x"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha") # real/double
    params["beta"] = init_real("beta") # real/double
    params["gamma"] = init_real("gamma") # real/double
    params["tau"] = init_real("tau", low=0) # real/double
    params["lambda_"] = init_vector("lambda_", dims=(Ndoses, Nplates)) # vector

def model(data, params):
    # initialize data
    Ndoses = data["Ndoses"]
    Nplates = data["Nplates"]
    y = data["y"]
    x = data["x"]
    # INIT parameters
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    tau = params["tau"]
    lambda_ = params["lambda_"]
    # initialize transformed parameters
    sigma = init_real("sigma", low=0) # real/double
    sigma = _pyro_assign(sigma, (1.0 / _call_func("sqrt", [tau])))
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0.0, 100])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0, 100])
    gamma =  _pyro_sample(gamma, "gamma", "normal", [0.0, 100000.0])
    tau =  _pyro_sample(tau, "tau", "gamma", [0.001, 0.001])
    for dose in range(1, to_int(Ndoses) + 1):

        lambda_[dose - 1] =  _pyro_sample(_index_select(lambda_, dose - 1) , "lambda_[%d]" % (to_int(dose-1)), "normal", [0.0, sigma])
        y[dose - 1] =  _pyro_sample(_index_select(y, dose - 1) , "y[%d]" % (to_int(dose-1)), "poisson_log", [_call_func("add", [((alpha + (beta * _call_func("log", [(_index_select(x, dose - 1)  + 10)]))) + (gamma * _index_select(x, dose - 1) )),_index_select(lambda_, dose - 1) ])], obs=_index_select(y, dose - 1) )

