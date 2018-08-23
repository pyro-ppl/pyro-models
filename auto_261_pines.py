# model file: ../example-models/bugs_examples/vol2/pines/pines.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'z' in data, 'variable not found in data: key=z'
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    z = data["z"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(y, dims=[N])
    check_constraints(x, dims=[N])
    check_constraints(z, dims=[N])

def transformed_data(data):
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    z = data["z"]
    y_std = init_vector("y_std", dims=(N)) # vector
    x_std = init_vector("x_std", dims=(N)) # vector
    z_std = init_vector("z_std", dims=(N)) # vector
    y_std = _pyro_assign(y_std, _call_func("divide", [_call_func("subtract", [y,_call_func("mean", [y])]),_call_func("sd", [y])]))
    x_std = _pyro_assign(x_std, _call_func("divide", [_call_func("subtract", [x,_call_func("mean", [x])]),_call_func("sd", [x])]))
    z_std = _pyro_assign(z_std, _call_func("divide", [_call_func("subtract", [z,_call_func("mean", [z])]),_call_func("sd", [z])]))
    data["y_std"] = y_std
    data["x_std"] = x_std
    data["z_std"] = z_std

def init_params(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    z = data["z"]
    # initialize transformed data
    y_std = data["y_std"]
    x_std = data["x_std"]
    z_std = data["z_std"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha") # real/double
    params["beta"] = init_real("beta") # real/double
    params["gamma"] = init_real("gamma") # real/double
    params["delta"] = init_real("delta") # real/double
    params["tau"] = init_vector("tau", low=0, dims=(2)) # vector
    params["lambda_"] = init_real("lambda_", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    x = data["x"]
    z = data["z"]
    # initialize transformed data
    y_std = data["y_std"]
    x_std = data["x_std"]
    z_std = data["z_std"]
    # INIT parameters
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    tau = params["tau"]
    lambda_ = params["lambda_"]
    # initialize transformed parameters
    sigma = init_vector("sigma", low=0, dims=(2)) # vector
    log_py = init_vector("log_py", dims=(2)) # vector
    for i in range(1, 2 + 1):
        sigma[i - 1] = _pyro_assign(sigma[i - 1], (1 / _call_func("sqrt", [_index_select(tau, i - 1) ])))
    log_py[1 - 1] = _pyro_assign(log_py[1 - 1], ((((((((_call_func("log", [lambda_]) + _call_func("log", [0.99950000000000006])) + _call_func("normal_log", [y_std,_call_func("add", [alpha,_call_func("multiply", [beta,x_std])]),_index_select(sigma, 1 - 1) ])) + _call_func("normal_log", [alpha,0,_call_func("sqrt", [1000000.0])])) + _call_func("normal_log", [beta,0,_call_func("sqrt", [10000.0])])) + _call_func("gamma_log", [_index_select(tau, 1 - 1) ,0.0001,0.0001])) + _call_func("normal_log", [gamma,0,_call_func("sqrt", [(1 / 400.0)])])) + _call_func("normal_log", [delta,1,_call_func("sqrt", [(1 / 400.0)])])) + _call_func("gamma_log", [_index_select(tau, 2 - 1) ,46,4.5])))
    log_py[2 - 1] = _pyro_assign(log_py[2 - 1], ((((((((_call_func("log", [lambda_]) + _call_func("log1m", [0.00050000000000000001])) + _call_func("normal_log", [y_std,_call_func("add", [gamma,_call_func("multiply", [delta,z_std])]),_index_select(sigma, 2 - 1) ])) + _call_func("normal_log", [gamma,0,_call_func("sqrt", [1000000.0])])) + _call_func("normal_log", [delta,0,_call_func("sqrt", [10000.0])])) + _call_func("gamma_log", [_index_select(tau, 2 - 1) ,0.0001,0.0001])) + _call_func("normal_log", [alpha,0,_call_func("sqrt", [(1 / 256.0)])])) + _call_func("normal_log", [beta,1,_call_func("sqrt", [(1 / 256.0)])])) + _call_func("gamma_log", [_index_select(tau, 1 - 1) ,30,4.5])))
    # model block

    pyro.sample("_call_func( log_sum_exp , [log_py])", dist.Bernoulli(_call_func("log_sum_exp", [log_py])), obs=(1));

