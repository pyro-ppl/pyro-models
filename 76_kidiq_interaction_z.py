# model file: ../example-models/ARM/Ch.4/kidiq_interaction_z.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'kid_score' in data, 'variable not found in data: key=kid_score'
    assert 'mom_hs' in data, 'variable not found in data: key=mom_hs'
    assert 'mom_iq' in data, 'variable not found in data: key=mom_iq'
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    z_mom_hs = init_vector("z_mom_hs", dims=(N)) # vector
    z_mom_iq = init_vector("z_mom_iq", dims=(N)) # vector
    inter = init_vector("inter", dims=(N)) # vector
    z_mom_hs = _pyro_assign(z_mom_hs, _call_func("divide", [_call_func("subtract", [mom_hs,_call_func("mean", [mom_hs])]),(2 * _call_func("sd", [mom_hs]))]))
    z_mom_iq = _pyro_assign(z_mom_iq, _call_func("divide", [_call_func("subtract", [mom_iq,_call_func("mean", [mom_iq])]),(2 * _call_func("sd", [mom_iq]))]))
    inter = _pyro_assign(inter, _call_func("elt_multiply", [z_mom_hs,z_mom_iq]))
    data["z_mom_hs"] = z_mom_hs
    data["z_mom_iq"] = z_mom_iq
    data["inter"] = inter

def init_params(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    # initialize transformed data
    z_mom_hs = data["z_mom_hs"]
    z_mom_iq = data["z_mom_iq"]
    inter = data["inter"]
    # assign init values for parameters
    params["beta"] = init_vector("beta", dims=(4)) # vector
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    # initialize transformed data
    z_mom_hs = data["z_mom_hs"]
    z_mom_iq = data["z_mom_iq"]
    inter = data["inter"]
    
    # init parameters
    beta = params["beta"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    kid_score =  _pyro_sample(kid_score, "kid_score", "normal", [_call_func("add", [_call_func("add", [_call_func("add", [_index_select(beta, 1 - 1) ,_call_func("multiply", [_index_select(beta, 2 - 1) ,z_mom_hs])]),_call_func("multiply", [_index_select(beta, 3 - 1) ,z_mom_iq])]),_call_func("multiply", [_index_select(beta, 4 - 1) ,inter])]), sigma], obs=kid_score)

