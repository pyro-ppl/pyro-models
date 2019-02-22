# model file: example-models/bugs_examples/vol1/lsat/lsat.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))



def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'R' in data, 'variable not found in data: key=R'
    assert 'T' in data, 'variable not found in data: key=T'
    assert 'culm' in data, 'variable not found in data: key=culm'
    assert 'response' in data, 'variable not found in data: key=response'
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]

def transformed_data(data):
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]
    with torch.no_grad():
        r = torch.empty(T, N)
        for j in range(int(culm[0])):
            for k in range(T):
                r[k, j] = response[0, k]
        for i in range(1, R):
            for j in range(int(culm[i-1]), int(culm[i])):
                for k in range(T):
                    r[k,j] = response[i, k]
    data["r"] = r

def init_params(data):
    params = {}
    return params

def model(data, params):
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]
    # initialize transformed data
    r = data["r"]
    # initialize transformed parameters
    # model block

    alpha =  pyro.sample("alpha", dist.Normal(0., 100.0))
    theta =  pyro.sample("theta", dist.Normal(0., 1))
    beta =  pyro.sample("beta", dist.Normal(0., 100.0))

    with pyro.plate('students', N):
        with pyro.plate('questions', T):
            r = pyro.sample('r', dist.Bernoulli(logits=beta * theta - alpha), obs=r)

