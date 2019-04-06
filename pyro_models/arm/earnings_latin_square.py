# model file: example-models/ARM/Ch.13/earnings_latin_square.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)).to_event(1))

def validate_data_def(data):
    assert 'N' in data, 'variable not found in data: key=N'
    assert 'n_age' in data, 'variable not found in data: key=n_age'
    assert 'n_eth' in data, 'variable not found in data: key=n_eth'
    assert 'age' in data, 'variable not found in data: key=age'
    assert 'eth' in data, 'variable not found in data: key=eth'
    assert 'x' in data, 'variable not found in data: key=x'
    assert 'y' in data, 'variable not found in data: key=y'
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"]
    eth = data["eth"]
    x = data["x"]
    y = data["y"]

def init_params(data):
    params = {}
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"]
    eth = data["eth"]
    x = data["x"]
    y = data["y"]
    # assign init values for parameters
    params["sigma_a1"] = pyro.sample("sigma_a1", dist.Uniform(0., 100.))
    params["sigma_a2"] = pyro.sample("sigma_a2", dist.Uniform(0., 100.))
    params["sigma_b1"] = pyro.sample("sigma_b1", dist.Uniform(0., 100.))
    params["sigma_b2"] = pyro.sample("sigma_b2", dist.Uniform(0., 100.))
    params["sigma_c"] = pyro.sample("sigma_c", dist.Uniform(0., 100.))
    params["sigma_d"] = pyro.sample("sigma_d", dist.Uniform(0., 100.))
    params["sigma_y"] = pyro.sample("sigma_y", dist.Uniform(0., 100.))

    return params

def model(data, params):
    # initialize data
    N = data["N"]
    n_age = data["n_age"]
    n_eth = data["n_eth"]
    age = data["age"].long() - 1
    eth = data["eth"].long() - 1
    x = data["x"]
    y = data["y"]

    # init parameters
    sigma_a1 = params["sigma_a1"]
    sigma_a2 = params["sigma_a2"]
    sigma_b1 = params["sigma_b1"]
    sigma_b2 = params["sigma_b2"]
    sigma_c = params["sigma_c"]
    sigma_d = params["sigma_d"]
    sigma_y = params["sigma_y"]

    mu_a1 = pyro.sample('mu_a1', dist.Normal(0., 1.))
    mu_a2 = pyro.sample('mu_a2', dist.Normal(0., 1.))
    mu_b1 = pyro.sample('mu_b1', dist.Normal(0., 1.))
    mu_b2 = pyro.sample('mu_b2', dist.Normal(0., 1.))
    mu_c = pyro.sample('mu_c', dist.Normal(0., 1.))
    mu_d = pyro.sample('mu_d', dist.Normal(0., 1.))
    plate_a = pyro.plate("as", n_eth, dim=-2)
    plate_b = pyro.plate("bs", n_age, dim=-1)

    with plate_a:
        a1 = pyro.sample('a1', dist.Normal(10 * mu_a1, sigma_a1))
        a2 = pyro.sample('a2', dist.Normal(10 * mu_a2, sigma_a2))
    with plate_b:
        b1 = pyro.sample('b1', dist.Normal(10 * mu_b1, sigma_b1))
        b2 = pyro.sample('b2', dist.Normal(0.1* mu_b2, sigma_b2))

    with plate_a, plate_b:
        c = pyro.sample('c', dist.Normal(10. * mu_c, sigma_c))
        d = pyro.sample('d', dist.Normal(0.1 * mu_d, sigma_d))

    with pyro.plate("data", N):
        y_hat = a1[..., eth, :].squeeze(-1) + a2[..., eth, :].squeeze(-1) * x + b1[..., age].squeeze() + b2[..., age].squeeze() * \
                x + c[..., eth, age] + d[..., eth, age] * x

        # A hack to make dimensions broadcast correctly when there is an IW plate
        if len(a1.size()) > 2:
            y_hat = y_hat.unsqueeze(-2)

        pyro.sample('y', dist.Normal(y_hat, sigma_y), obs=y)
