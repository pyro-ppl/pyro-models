# model file: ../example-models/bugs_examples/vol1/salm/salm.stan
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
    check_constraints(x, dims=[Ndoses])

def transformed_data(data):
    # initialize data
    Ndoses = data["Ndoses"]
    Nplates = data["Nplates"]
    y = data["y"]
    x = data["x"]
    logx = init_real("logx", dims=(Ndoses)) # real/double
    mean_x = init_real("mean_x") # real/double
    mean_logx = init_real("mean_logx") # real/double
    centered_x = init_real("centered_x", dims=(Ndoses)) # real/double
    centered_logx = init_real("centered_logx", dims=(Ndoses)) # real/double
    mean_x = _pyro_assign(mean_x, _call_func("mean", [x]))
    for dose in range(1, to_int(Ndoses) + 1):
        centered_x[dose - 1] = _pyro_assign(centered_x[dose - 1], (_index_select(x, dose - 1)  - mean_x))
    for dose in range(1, to_int(Ndoses) + 1):
        logx[dose - 1] = _pyro_assign(logx[dose - 1], _call_func("log", [(_index_select(x, dose - 1)  + 10)]))
    mean_logx = _pyro_assign(mean_logx, _call_func("mean", [logx]))
    for dose in range(1, to_int(Ndoses) + 1):
        centered_logx[dose - 1] = _pyro_assign(centered_logx[dose - 1], (_index_select(logx, dose - 1)  - mean_logx))
    data["logx"] = logx
    data["mean_x"] = mean_x
    data["mean_logx"] = mean_logx
    data["centered_x"] = centered_x
    data["centered_logx"] = centered_logx

def init_params(data, params):
    # initialize data
    Ndoses = data["Ndoses"]
    Nplates = data["Nplates"]
    y = data["y"]
    x = data["x"]
    # initialize transformed data
    logx = data["logx"]
    mean_x = data["mean_x"]
    mean_logx = data["mean_logx"]
    centered_x = data["centered_x"]
    centered_logx = data["centered_logx"]
    # assign init values for parameters
    params["alpha_star"] = init_real("alpha_star") # real/double
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
    # initialize transformed data
    logx = data["logx"]
    mean_x = data["mean_x"]
    mean_logx = data["mean_logx"]
    centered_x = data["centered_x"]
    centered_logx = data["centered_logx"]
    # INIT parameters
    alpha_star = params["alpha_star"]
    beta = params["beta"]
    gamma = params["gamma"]
    tau = params["tau"]
    lambda_ = params["lambda_"]
    # initialize transformed parameters
    sigma = init_real("sigma", low=0) # real/double
    alpha = init_real("alpha") # real/double
    alpha = _pyro_assign(alpha, ((alpha_star - (beta * mean_logx)) - (gamma * mean_x)))
    sigma = _pyro_assign(sigma, (1.0 / _call_func("sqrt", [tau])))
    # model block

    alpha_star =  _pyro_sample(alpha_star, "alpha_star", "normal", [0.0, 1000.0])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0, 1000])
    gamma =  _pyro_sample(gamma, "gamma", "normal", [0.0, 1000])
    tau =  _pyro_sample(tau, "tau", "gamma", [0.001, 0.001])
    for dose in range(1, to_int(Ndoses) + 1):

        lambda_[dose - 1] =  _pyro_sample(_index_select(lambda_, dose - 1) , "lambda_[%d]" % (to_int(dose-1)), "normal", [0.0, sigma])
        y[dose - 1] =  _pyro_sample(_index_select(y, dose - 1) , "y[%d]" % (to_int(dose-1)), "poisson_log", [_call_func("add", [((alpha_star + (beta * _index_select(centered_logx, dose - 1) )) + (gamma * _index_select(centered_x, dose - 1) )),_index_select(lambda_, dose - 1) ])], obs=_index_select(y, dose - 1) )

