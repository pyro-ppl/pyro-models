# model file: example-models/ARM/Ch.4/kidscore_momwork.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'kid_score' in data, 'variable not found in data: key=kid_score'
    assert 'mom_work' in data, 'variable not found in data: key=mom_work'
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_work = data["mom_work"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_work = data["mom_work"]
    work2 = mom_work == 2
    work3 = mom_work == 3
    work4 = mom_work == 4
    data["work2"] = work2
    data["work3"] = work3
    data["work4"] = work4

def init_params(data):
    params = {}
    params["beta"] = init_vector("beta", dims=(4)) # vector
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    kid_score = data["kid_score"]
    mom_work = data["mom_work"]
    # initialize transformed data
    work2 = data["work2"].float()
    work3 = data["work3"].float()
    work4 = data["work4"].float()

    # init parameters
    beta = params["beta"]
    sigma =  pyro.sample("sigma", dist.HalfCauchy(torch.tensor(2.5)))
    with pyro.plate("data", N):
        kid_score = pyro.sample('obs', dist.Normal(beta[0] + beta[1] * work2 + beta[2] * work3 + beta[3] * work4, sigma), obs=kid_score)
