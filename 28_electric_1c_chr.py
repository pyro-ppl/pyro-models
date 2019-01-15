# model file: ../example-models/ARM/Ch.23/electric_1c_chr.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_grade' in data, 'variable not found in data: key=n_grade'
    assert 'n_grade_pair' in data, 'variable not found in data: key=n_grade_pair'
    assert 'n_pair' in data, 'variable not found in data: key=n_pair'
    assert 'grade' in data, 'variable not found in data: key=grade'
    assert 'grade_pair' in data, 'variable not found in data: key=grade_pair'
    assert 'pair' in data, 'variable not found in data: key=pair'
    assert 'pre_test' in data, 'variable not found in data: key=pre_test'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_grade = data["n_grade"]
    n_grade_pair = data["n_grade_pair"]
    n_pair = data["n_pair"]
    grade = data["grade"]
    grade_pair = data["grade_pair"]
    pair = data["pair"]
    pre_test = data["pre_test"]
    treatment = data["treatment"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    n_grade = data["n_grade"]
    n_grade_pair = data["n_grade_pair"]
    n_pair = data["n_pair"]
    grade = data["grade"]
    grade_pair = data["grade_pair"]
    pair = data["pair"]
    pre_test = data["pre_test"]
    treatment = data["treatment"]
    y = data["y"]
    # assign init values for parameters
    params["sigma_a"] = pyro.sample("sigma_a", dist.Uniform(0., 100.).expand([n_grade_pair])) # vector
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.).expand([n_grade])) # vector

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_grade = data["n_grade"]
    n_grade_pair = data["n_grade_pair"]
    n_pair = data["n_pair"]
    grade = data["grade"].long() - 1
    grade_pair = data["grade_pair"].long() - 1
    pair = data["pair"].long() - 1
    pre_test = data["pre_test"]
    treatment = data["treatment"]
    y = data["y"]

    # init parameters
    sigma_a = params["sigma_a"]
    sigma_y = params["sigma_y"]

    # model block
    eta_a =  pyro.sample("eta_a", dist.Normal(0., 1.).expand([n_pair]))
    mu_a =  pyro.sample("mu_a", dist.Normal(0., 1.).expand([n_grade_pair]))
    a = 50 * mu_a[grade_pair] + sigma_a[grade_pair] * eta_a
    b = pyro.sample("b", dist.Normal(0., 100.).expand([n_grade]))
    c = pyro.sample("c", dist.Normal(0., 100.).expand([n_grade]))
    y_hat = a[pair] + b[grade] * treatment + c[grade] * pre_test
    sigma_y_hat = sigma_y[grade]
    y =  pyro.sample("y", dist.Normal(y_hat, sigma_y_hat), obs=y)
