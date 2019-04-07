# model file: example-models/ARM/Ch.23/electric_1a.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_pair' in data, 'variable not found in data: key=n_pair'
    assert 'n_grade' in data, 'variable not found in data: key=n_grade'
    assert 'n_grade_pair' in data, 'variable not found in data: key=n_grade_pair'
    assert 'grade' in data, 'variable not found in data: key=grade'
    assert 'grade_pair' in data, 'variable not found in data: key=grade_pair'
    assert 'pair' in data, 'variable not found in data: key=pair'
    assert 'treatment' in data, 'variable not found in data: key=treatment'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    n_grade = data["n_grade"]
    n_grade_pair = data["n_grade_pair"]
    grade = data["grade"]
    grade_pair = data["grade_pair"]
    pair = data["pair"]
    treatment = data["treatment"]
    y = data["y"]

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_pair = data["n_pair"]
    n_grade = data["n_grade"]
    n_grade_pair = data["n_grade_pair"]
    grade = data["grade"].long() - 1
    grade_pair = data["grade_pair"].long() - 1
    pair = data["pair"].long() - 1
    treatment = data["treatment"]
    y = data["y"]

    # model block
    with pyro.plate('n_grade_pair', n_grade_pair):
        mu_a =  pyro.sample("mu_a", dist.Normal(0., 1.))
        sigma_a = pyro.sample("sigma_a", dist.Uniform(0., 100.))
    sigma_a_hat = sigma_a[...,grade_pair]
    mu_a_hat = 100 * mu_a[...,grade_pair]
    with pyro.plate('n_pair', n_pair):
        a =  pyro.sample("a", dist.Normal(mu_a_hat, sigma_a_hat))
    with pyro.plate('n_grade', n_grade):
        b = pyro.sample("b", dist.Normal(0., 100.))
        sigma_y = pyro.sample("sigma_y", dist.Uniform(0., 100.))
    sigma_y_hat = sigma_y[...,grade]
    with pyro.plate("data", N):
        y_hat = a[...,pair] + b[...,grade] * treatment
        y =  pyro.sample("y", dist.Normal(y_hat, sigma_y_hat), obs=y)
