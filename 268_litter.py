# model file: ../example-models/bugs_examples/vol1/litter/litter.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'G' in data, 'variable not found in data: key=G'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'r' in data, 'variable not found in data: key=r'
    assert 'n' in data, 'variable not found in data: key=n'
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]

def init_params(data, params):
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]
    # assign init values for parameters
    params["p"] = init_matrix("p", dist.Uniform(0., 1, dims=(G, N)) # matrix
    params["mu"] = init_vector("mu", dist.Uniform(0., 1, dims=(G)) # vector
    params["a_plus_b"] = init_vector("a_plus_b", dist.Uniform(0.10000000000000001, dims=(G)) # vector

def model(data, params):
    # initialize data
    G = data["G"]
    N = data["N"]
    r = data["r"]
    n = data["n"]
    
    # init parameters
    p = params["p"]
    mu = params["mu"]
    a_plus_b = params["a_plus_b"]
    # initialize transformed parameters
    a = init_vector("a", dims=(G)) # vector
    b = init_vector("b", dims=(G)) # vector
    a = _pyro_assign(a, _call_func("elt_multiply", [mu,a_plus_b]))
    b = _pyro_assign(b, _call_func("elt_multiply", [_call_func("subtract", [1,mu]),a_plus_b]))
    # model block

    a_plus_b =  _pyro_sample(a_plus_b, "a_plus_b", "pareto", [0.10000000000000001, 1.5])
    for g in range(1, to_int(G) + 1):

        for i in range(1, to_int(N) + 1):

            p[g - 1][i - 1] =  _pyro_sample(_index_select(_index_select(p, g - 1) , i - 1) , "p[%d][%d]" % (to_int(g-1),to_int(i-1)), "beta", [_index_select(a, g - 1) , _index_select(b, g - 1) ])
            r[g - 1][i - 1] =  _pyro_sample(_index_select(_index_select(r, g - 1) , i - 1) , "r[%d][%d]" % (to_int(g-1),to_int(i-1)), "binomial", [_index_select(_index_select(n, g - 1) , i - 1) , _index_select(_index_select(p, g - 1) , i - 1) ], obs=_index_select(_index_select(r, g - 1) , i - 1) )

