# model file: ../example-models/ARM/Ch.18/weight_censored.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'N_obs' in data, 'variable not found in data: key=N_obs'
    assert 'N_cens' in data, 'variable not found in data: key=N_cens'
    assert 'weight' in data, 'variable not found in data: key=weight'
    assert 'height' in data, 'variable not found in data: key=height'
    assert 'C' in data, 'variable not found in data: key=C'
    # initialize data
    N = data["N"]
    N_obs = data["N_obs"]
    N_cens = data["N_cens"]
    weight = data["weight"]
    height = data["height"]
    C = data["C"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    N_obs = data["N_obs"]
    N_cens = data["N_cens"]
    weight = data["weight"]
    height = data["height"]
    C = data["C"]
    c_height = init_vector("c_height", dims=(N)) # vector
    weight_obs = init_vector("weight_obs", dims=(N_obs)) # vector
    c_height_obs = init_vector("c_height_obs", dims=(N_obs)) # vector
    c_height_cens = init_vector("c_height_cens", dims=(N_cens)) # vector
    i = init_int("i"))
    j = init_int("j"))
    c_height = _pyro_assign(c_height, _call_func("subtract", [height,_call_func("mean", [height])]))
    i = _pyro_assign(i, 1)
    j = _pyro_assign(j, 1)
    for n in range(1, to_int(N) + 1):

        if (as_bool(_call_func("logical_eq", [_index_select(weight, n - 1) ,C]))):

            c_height_cens[i - 1] = _pyro_assign(c_height_cens[i - 1], _index_select(c_height, n - 1) )
            i = _pyro_assign(i, (i + 1))
        else: 

            weight_obs[j - 1] = _pyro_assign(weight_obs[j - 1], _index_select(weight, n - 1) )
            c_height_obs[j - 1] = _pyro_assign(c_height_obs[j - 1], _index_select(c_height, n - 1) )
            j = _pyro_assign(j, (j + 1))
        
    data["c_height"] = c_height
    data["weight_obs"] = weight_obs
    data["c_height_obs"] = c_height_obs
    data["c_height_cens"] = c_height_cens
    data["i"] = i
    data["j"] = j

def init_params(data, params):
    # initialize data
    N = data["N"]
    N_obs = data["N_obs"]
    N_cens = data["N_cens"]
    weight = data["weight"]
    height = data["height"]
    C = data["C"]
    # initialize transformed data
    c_height = data["c_height"]
    weight_obs = data["weight_obs"]
    c_height_obs = data["c_height_obs"]
    c_height_cens = data["c_height_cens"]
    i = data["i"]
    j = data["j"]
    # assign init values for parameters
    params["weight_cens"] = init_vector("weight_cens", dist.Uniform(C, dims=(N_cens)) # vector
    params["a"] = pyro.sample("a"))
    params["b"] = pyro.sample("b"))
    params["sigma"] = pyro.sample("sigma", dist.Uniform(0))

def model(data, params):
    # initialize data
    N = data["N"]
    N_obs = data["N_obs"]
    N_cens = data["N_cens"]
    weight = data["weight"]
    height = data["height"]
    C = data["C"]
    # initialize transformed data
    c_height = data["c_height"]
    weight_obs = data["weight_obs"]
    c_height_obs = data["c_height_obs"]
    c_height_cens = data["c_height_cens"]
    i = data["i"]
    j = data["j"]
    
    # init parameters
    weight_cens = params["weight_cens"]
    a = params["a"]
    b = params["b"]
    sigma = params["sigma"]
    # initialize transformed parameters
    # model block

    weight_obs =  _pyro_sample(weight_obs, "weight_obs", "normal", [_call_func("add", [a,_call_func("multiply", [b,c_height_obs])]), sigma], obs=weight_obs)
    weight_cens =  _pyro_sample(weight_cens, "weight_cens", "normal", [_call_func("add", [a,_call_func("multiply", [b,c_height_cens])]), sigma])

