# model file: ../example-models/misc/irt/irt_multilevel.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'K' in data, 'variable not found in data: key=K'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'jj' in data, 'variable not found in data: key=jj'
    assert 'kk' in data, 'variable not found in data: key=kk'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"]
    kk = data["kk"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"]
    kk = data["kk"]
    y = data["y"]
    # assign init values for parameters
    params["delta"] = pyro.sample("delta"))
    params["alpha"] = pyro.sample("alpha", dims=(J)))
    params["beta"] = pyro.sample("beta", dims=(K)))
    params["sigma_alpha"] = pyro.sample("sigma_alpha", dist.Uniform(0))
    params["sigma_beta"] = pyro.sample("sigma_beta", dist.Uniform(0))

    return params

def model(data, params):
    # initialize data
    J = data["J"]
    K = data["K"]
    N = data["N"]
    jj = data["jj"]
    kk = data["kk"]
    y = data["y"]
    
    # init parameters
    delta = params["delta"]
    alpha = params["alpha"]
    beta = params["beta"]
    sigma_alpha = params["sigma_alpha"]
    sigma_beta = params["sigma_beta"]
    # initialize transformed parameters
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0., sigma_alpha])
    beta =  _pyro_sample(beta, "beta", "normal", [0., sigma_beta])
    delta =  _pyro_sample(delta, "delta", "normal", [0.75, 1])
    for n in range(1, to_int(N) + 1):
        y[n - 1] =  _pyro_sample(_index_select(y, n - 1) , "y[%d]" % (to_int(n-1)), "bernoulli_logit", [((_index_select(alpha, jj[n - 1] - 1)  - _index_select(beta, kk[n - 1] - 1) ) + delta)], obs=_index_select(y, n - 1) )

