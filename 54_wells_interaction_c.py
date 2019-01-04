# model file: ../example-models/ARM/Ch.5/wells_interaction_c.stan
import torch
import pyro


def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'arsenic' in data, 'variable not found in data: key=arsenic'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    c_dist100 = init_vector("c_dist100", dims=(N)) # vector
    c_arsenic = init_vector("c_arsenic", dims=(N)) # vector
    inter = init_vector("inter", dims=(N)) # vector
    c_dist100 = _pyro_assign(c_dist100, _call_func("divide", [_call_func("subtract", [dist,_call_func("mean", [dist])]),100.0]))
    c_arsenic = _pyro_assign(c_arsenic, _call_func("subtract", [arsenic,_call_func("mean", [arsenic])]))
    inter = _pyro_assign(inter, _call_func("elt_multiply", [c_dist100,c_arsenic]))
    data["c_dist100"] = c_dist100
    data["c_arsenic"] = c_arsenic
    data["inter"] = inter

def init_params(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    # initialize transformed data
    c_dist100 = data["c_dist100"]
    c_arsenic = data["c_arsenic"]
    inter = data["inter"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    # initialize transformed data
    c_dist100 = data["c_dist100"]
    c_arsenic = data["c_arsenic"]
    inter = data["inter"]
    # INIT parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    switched =  _pyro_sample(switched, "switched", "bernoulli_logit", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,c_dist100])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,c_arsenic])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,inter])])], obs=switched)

