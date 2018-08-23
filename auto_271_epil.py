# model file: ../example-models/bugs_examples/vol1/epil/epil.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'Trt' in data, 'variable not found in data: key=Trt'
    assert 'V4' in data, 'variable not found in data: key=V4'
    assert 'log_Base4' in data, 'variable not found in data: key=log_Base4'
    assert 'log_Age' in data, 'variable not found in data: key=log_Age'
    assert 'BT' in data, 'variable not found in data: key=BT'
    assert 'log_Age_bar' in data, 'variable not found in data: key=log_Age_bar'
    assert 'Trt_bar' in data, 'variable not found in data: key=Trt_bar'
    assert 'BT_bar' in data, 'variable not found in data: key=BT_bar'
    assert 'V4_bar' in data, 'variable not found in data: key=V4_bar'
    assert 'log_Base4_bar' in data, 'variable not found in data: key=log_Base4_bar'
    # initialize data
    N = data["N"]
    T = data["T"]
    y = data["y"]
    Trt = data["Trt"]
    V4 = data["V4"]
    log_Base4 = data["log_Base4"]
    log_Age = data["log_Age"]
    BT = data["BT"]
    log_Age_bar = data["log_Age_bar"]
    Trt_bar = data["Trt_bar"]
    BT_bar = data["BT_bar"]
    V4_bar = data["V4_bar"]
    log_Base4_bar = data["log_Base4_bar"]
    check_constraints(N, low=0, dims=[1])
    check_constraints(T, low=0, dims=[1])
    check_constraints(y, low=0, dims=[N, T])
    check_constraints(Trt, low=0, dims=[N])
    check_constraints(V4, low=0, dims=[T])
    check_constraints(log_Base4, dims=[N])
    check_constraints(log_Age, dims=[N])
    check_constraints(BT, dims=[N])
    check_constraints(log_Age_bar, dims=[1])
    check_constraints(Trt_bar, dims=[1])
    check_constraints(BT_bar, dims=[1])
    check_constraints(V4_bar, dims=[1])
    check_constraints(log_Base4_bar, dims=[1])

def transformed_data(data):
    # initialize data
    N = data["N"]
    T = data["T"]
    y = data["y"]
    Trt = data["Trt"]
    V4 = data["V4"]
    log_Base4 = data["log_Base4"]
    log_Age = data["log_Age"]
    BT = data["BT"]
    log_Age_bar = data["log_Age_bar"]
    Trt_bar = data["Trt_bar"]
    BT_bar = data["BT_bar"]
    V4_bar = data["V4_bar"]
    log_Base4_bar = data["log_Base4_bar"]
    V4_c = init_vector("V4_c", dims=(T)) # vector
    log_Base4_c = init_vector("log_Base4_c", dims=(N)) # vector
    log_Age_c = init_vector("log_Age_c", dims=(N)) # vector
    BT_c = init_vector("BT_c", dims=(N)) # vector
    Trt_c = init_vector("Trt_c", dims=(N)) # vector
    log_Base4_c = _pyro_assign(log_Base4_c, _call_func("subtract", [log_Base4,log_Base4_bar]))
    log_Age_c = _pyro_assign(log_Age_c, _call_func("subtract", [log_Age,log_Age_bar]))
    BT_c = _pyro_assign(BT_c, _call_func("subtract", [BT,BT_bar]))
    for i in range(1, to_int(T) + 1):
        V4_c[i - 1] = _pyro_assign(V4_c[i - 1], (_index_select(V4, i - 1)  - V4_bar))
    for i in range(1, to_int(N) + 1):
        Trt_c[i - 1] = _pyro_assign(Trt_c[i - 1], (_index_select(Trt, i - 1)  - Trt_bar))
    data["V4_c"] = V4_c
    data["log_Base4_c"] = log_Base4_c
    data["log_Age_c"] = log_Age_c
    data["BT_c"] = BT_c
    data["Trt_c"] = Trt_c

def init_params(data, params):
    # initialize data
    N = data["N"]
    T = data["T"]
    y = data["y"]
    Trt = data["Trt"]
    V4 = data["V4"]
    log_Base4 = data["log_Base4"]
    log_Age = data["log_Age"]
    BT = data["BT"]
    log_Age_bar = data["log_Age_bar"]
    Trt_bar = data["Trt_bar"]
    BT_bar = data["BT_bar"]
    V4_bar = data["V4_bar"]
    log_Base4_bar = data["log_Base4_bar"]
    # initialize transformed data
    V4_c = data["V4_c"]
    log_Base4_c = data["log_Base4_c"]
    log_Age_c = data["log_Age_c"]
    BT_c = data["BT_c"]
    Trt_c = data["Trt_c"]
    # assign init values for parameters
    params["a0"] = init_real("a0") # real/double
    params["alpha_Base"] = init_real("alpha_Base") # real/double
    params["alpha_Trt"] = init_real("alpha_Trt") # real/double
    params["alpha_BT"] = init_real("alpha_BT") # real/double
    params["alpha_Age"] = init_real("alpha_Age") # real/double
    params["alpha_V4"] = init_real("alpha_V4") # real/double
    params["b1"] = init_real("b1", dims=(N)) # real/double
    params["b"] = init_vector("b", dims=(N, T)) # vector
    params["sigmasq_b"] = init_real("sigmasq_b", low=0) # real/double
    params["sigmasq_b1"] = init_real("sigmasq_b1", low=0) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    T = data["T"]
    y = data["y"]
    Trt = data["Trt"]
    V4 = data["V4"]
    log_Base4 = data["log_Base4"]
    log_Age = data["log_Age"]
    BT = data["BT"]
    log_Age_bar = data["log_Age_bar"]
    Trt_bar = data["Trt_bar"]
    BT_bar = data["BT_bar"]
    V4_bar = data["V4_bar"]
    log_Base4_bar = data["log_Base4_bar"]
    # initialize transformed data
    V4_c = data["V4_c"]
    log_Base4_c = data["log_Base4_c"]
    log_Age_c = data["log_Age_c"]
    BT_c = data["BT_c"]
    Trt_c = data["Trt_c"]
    # INIT parameters
    a0 = params["a0"]
    alpha_Base = params["alpha_Base"]
    alpha_Trt = params["alpha_Trt"]
    alpha_BT = params["alpha_BT"]
    alpha_Age = params["alpha_Age"]
    alpha_V4 = params["alpha_V4"]
    b1 = params["b1"]
    b = params["b"]
    sigmasq_b = params["sigmasq_b"]
    sigmasq_b1 = params["sigmasq_b1"]
    # initialize transformed parameters
    sigma_b = init_real("sigma_b", low=0) # real/double
    sigma_b1 = init_real("sigma_b1", low=0) # real/double
    sigma_b = _pyro_assign(sigma_b, _call_func("sqrt", [sigmasq_b]))
    sigma_b1 = _pyro_assign(sigma_b1, _call_func("sqrt", [sigmasq_b1]))
    # model block

    a0 =  _pyro_sample(a0, "a0", "normal", [0, 100])
    alpha_Base =  _pyro_sample(alpha_Base, "alpha_Base", "normal", [0, 100])
    alpha_Trt =  _pyro_sample(alpha_Trt, "alpha_Trt", "normal", [0, 100])
    alpha_BT =  _pyro_sample(alpha_BT, "alpha_BT", "normal", [0, 100])
    alpha_Age =  _pyro_sample(alpha_Age, "alpha_Age", "normal", [0, 100])
    alpha_V4 =  _pyro_sample(alpha_V4, "alpha_V4", "normal", [0, 100])
    sigmasq_b1 =  _pyro_sample(sigmasq_b1, "sigmasq_b1", "inv_gamma", [0.001, 0.001])
    sigmasq_b =  _pyro_sample(sigmasq_b, "sigmasq_b", "inv_gamma", [0.001, 0.001])
    b1 =  _pyro_sample(b1, "b1", "normal", [0, sigma_b1])
    for n in range(1, to_int(N) + 1):

        b[n - 1] =  _pyro_sample(_index_select(b, n - 1) , "b[%d]" % (to_int(n-1)), "normal", [0, sigma_b])
        y[n - 1] =  _pyro_sample(_index_select(y, n - 1) , "y[%d]" % (to_int(n-1)), "poisson_log", [_call_func("add", [_call_func("add", [(((((a0 + (alpha_Base * _index_select(log_Base4_c, n - 1) )) + (alpha_Trt * _index_select(Trt_c, n - 1) )) + (alpha_BT * _index_select(BT_c, n - 1) )) + (alpha_Age * _index_select(log_Age_c, n - 1) )) + _index_select(b1, n - 1) ),_call_func("multiply", [alpha_V4,V4_c])]),_index_select(b, n - 1) ])], obs=_index_select(y, n - 1) )

