# model file: ../example-models/ARM/Ch.5/wells_predicted_log.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'switched' in data, 'variable not found in data: key=switched'
    assert 'dist' in data, 'variable not found in data: key=dist'
    assert 'arsenic' in data, 'variable not found in data: key=arsenic'
    assert 'educ' in data, 'variable not found in data: key=educ'
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    c_dist100 = init_vector("c_dist100", dims=(N)) # vector
    log_arsenic = init_vector("log_arsenic", dims=(N)) # vector
    c_log_arsenic = init_vector("c_log_arsenic", dims=(N)) # vector
    c_educ4 = init_vector("c_educ4", dims=(N)) # vector
    da_inter = init_vector("da_inter", dims=(N)) # vector
    de_inter = init_vector("de_inter", dims=(N)) # vector
    ae_inter = init_vector("ae_inter", dims=(N)) # vector
    c_dist100 = _pyro_assign(c_dist100., _call_func("divide", [_call_func("subtract", [dist,_call_func("mean", [dist])]),100.0]))
    log_arsenic = _pyro_assign(log_arsenic, _call_func("log", [arsenic]))
    c_log_arsenic = _pyro_assign(c_log_arsenic, _call_func("subtract", [log_arsenic,_call_func("mean", [log_arsenic])]))
    c_educ4 = _pyro_assign(c_educ4, _call_func("divide", [_call_func("subtract", [educ,_call_func("mean", [educ])]),4.0]))
    da_inter = _pyro_assign(da_inter, _call_func("elt_multiply", [c_dist100.,c_log_arsenic]))
    de_inter = _pyro_assign(de_inter, _call_func("elt_multiply", [c_dist100.,c_educ4]))
    ae_inter = _pyro_assign(ae_inter, _call_func("elt_multiply", [c_log_arsenic,c_educ4]))
    data["c_dist100"] = c_dist100
    data["log_arsenic"] = log_arsenic
    data["c_log_arsenic"] = c_log_arsenic
    data["c_educ4"] = c_educ4
    data["da_inter"] = da_inter
    data["de_inter"] = de_inter
    data["ae_inter"] = ae_inter

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    # initialize transformed data
    c_dist100 = data["c_dist100"]
    log_arsenic = data["log_arsenic"]
    c_log_arsenic = data["c_log_arsenic"]
    c_educ4 = data["c_educ4"]
    da_inter = data["da_inter"]
    de_inter = data["de_inter"]
    ae_inter = data["ae_inter"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(7)) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    switched = data["switched"]
    dist = data["dist"]
    arsenic = data["arsenic"]
    educ = data["educ"]
    # initialize transformed data
    c_dist100 = data["c_dist100"]
    log_arsenic = data["log_arsenic"]
    c_log_arsenic = data["c_log_arsenic"]
    c_educ4 = data["c_educ4"]
    da_inter = data["da_inter"]
    de_inter = data["de_inter"]
    ae_inter = data["ae_inter"]
    
    # init parameters
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    switched =  _pyro_sample(switched, "switched", "bernoulli_logit", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,c_dist100])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,c_log_arsenic])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,c_educ4])]),_call_func("multiply", [_index_select(beta, 5 - 1) ,da_inter])]),_call_func("multiply", [_index_select(beta, 6 - 1) ,de_inter])]),_call_func("multiply", [_index_select(beta, 7 - 1) ,ae_inter])])], obs=switched)

