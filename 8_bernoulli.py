# model file: ../example-models/basic_estimators/bernoulli.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    y = data["y"]

def init_params(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # assign init values for parameters
    params["theta"] = init_real("theta", low=0, high=1) # real/double

def model(data, params):
    # initialize data
    N = data["N"]
    y = data["y"]
    # INIT parameters
    theta = params["theta"]
    # initialize transformed parameters
    # model block

    theta =  _pyro_sample(theta, "theta", "beta", [1, 1])
    for n in range(1, to_int(N) + 1):
        y[n - 1] =  _pyro_sample(_index_select(y, n - 1) , "y[%d]" % (to_int(n-1)), "bernoulli", [theta], obs=_index_select(y, n - 1) )

