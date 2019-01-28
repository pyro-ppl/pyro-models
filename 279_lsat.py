# model file: ../example-models/bugs_examples/vol1/lsat/lsat.stan
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
    r = init_int("r", dims=(T, N)))
    ones = init_vector("ones", dims=(N)) # vector
    for j in range(1, to_int(culm[1 - 1]) + 1):

        for k in range(1, to_int(T) + 1):

            r[k - 1][j - 1] = _pyro_assign(r[k - 1][j - 1], _index_select(_index_select(response, 1 - 1) , k - 1) )
    for i in range(2, to_int(R) + 1):

        for j in range(to_int((culm[(i - 1) - 1] + 1)), to_int(culm[i - 1]) + 1):

            for k in range(1, to_int(T) + 1):

                r[k - 1][j - 1] = _pyro_assign(r[k - 1][j - 1], _index_select(_index_select(response, i - 1) , k - 1) )
    for i in range(1, to_int(N) + 1):
        ones[i - 1] = _pyro_assign(ones[i - 1], 1.0)
    data["r"] = r
    data["ones"] = ones

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    R = data["R"]
    T = data["T"]
    culm = data["culm"]
    response = data["response"]
    # initialize transformed data
    r = data["r"]
    ones = data["ones"]
    # assign init values for parameters
    params["alpha"] = pyro.sample("alpha", dims=(T)))
    params["theta"] = init_vector("theta", dims=(N)) # vector
    params["beta"] = pyro.sample("beta", dist.Uniform(0))

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
    ones = data["ones"]
    
    # init parameters
    alpha = params["alpha"]
    theta = params["theta"]
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0., 100.0])
    theta =  _pyro_sample(theta, "theta", "normal", [0., 1])
    beta =  _pyro_sample(beta, "beta", "normal", [0.0., 100.0])
    for k in range(1, to_int(T) + 1):
        r[k - 1] =  _pyro_sample(_index_select(r, k - 1) , "r[%d]" % (to_int(k-1)), "bernoulli_logit", [_call_func("subtract", [_call_func("multiply", [beta,theta]),_call_func("multiply", [_index_select(alpha, k - 1) ,ones])])], obs=_index_select(r, k - 1) )

