# model file: ../example-models/ARM/Ch.20/hiv_inter_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'person' in data, 'variable not found in data: key=person'
    assert 'time' in data, 'variable not found in data: key=time'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    treatment = data["treatment"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    treatment = data["treatment"]
    y = data["y"]
    # assign init values for parameters
    params["beta"] = pyro.sample("beta"))
    params["eta1"] = init_vector("eta1", dims=(J)) # vector
    params["eta2"] = init_vector("eta2", dims=(J)) # vector
    params["mu_a1"] = pyro.sample("mu_a1"))
    params["mu_a2"] = pyro.sample("mu_a2"))
    params["sigma_a1"] = pyro.sample("sigma_a1", dist.Uniform(0., 100.))
    params["sigma_a2"] = pyro.sample("sigma_a2", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    J = data["J"]
    N = data["N"]
    person = data["person"]
    time = data["time"]
    treatment = data["treatment"]
    y = data["y"]
    
    # init parameters
    beta = params["beta"]
    eta1 = params["eta1"]
    eta2 = params["eta2"]
    mu_a1 = params["mu_a1"]
    mu_a2 = params["mu_a2"]
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_y = params["sigma_y"]
    # initialize transformed parameters
    a1 = init_vector("a1", dims=(J)) # vector
    a2 = init_vector("a2", dims=(J)) # vector
    y_hat = init_vector("y_hat", dims=(N)) # vector
    a1 = _pyro_assign(a1, _call_func("add", [(10 * mu_a1),_call_func("multiply", [sigma_a1,eta1])]))
    a2 = _pyro_assign(a2, _call_func("add", [(0.10000000000000001 * mu_a2),_call_func("multiply", [sigma_a2,eta2])]))
    for i in range(1, to_int(N) + 1):
        y_hat[i - 1] = _pyro_assign(y_hat[i - 1], ((((beta * _index_select(time, i - 1) ) * _index_select(treatment, i - 1) ) + _index_select(a1, person[i - 1] - 1) ) + (_index_select(a2, person[i - 1] - 1)  * _index_select(time, i - 1) )))
    # model block

    mu_a1 =  _pyro_sample(mu_a1, "mu_a1", "normal", [0., 1])
    eta1 =  _pyro_sample(eta1, "eta1", "normal", [0., 1])
    mu_a2 =  _pyro_sample(mu_a2, "mu_a2", "normal", [0., 1])
    eta2 =  _pyro_sample(eta2, "eta2", "normal", [0., 1])
    beta =  _pyro_sample(beta, "beta", "normal", [0., 1])
    y =  _pyro_sample(y, "y", "normal", [y_hat, sigma_y], obs=y)

