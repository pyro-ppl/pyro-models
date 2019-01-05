# model file: ../example-models/bugs_examples/vol2/cervix/cervix2.stan
import torch
import pyro
import pyro.distributions as dist

def init_vector(name, dims=None):
    return pyro.sample(name, dist.Normal(torch.zeros(dims), 0.2 * torch.ones(dims)))



def validate_data_def(data):
    assert 'Nc' in data, 'variable not found in data: key=Nc'
    assert 'Ni' in data, 'variable not found in data: key=Ni'
    assert 'xc' in data, 'variable not found in data: key=xc'
    assert 'wc' in data, 'variable not found in data: key=wc'
    assert 'dc' in data, 'variable not found in data: key=dc'
    assert 'wi' in data, 'variable not found in data: key=wi'
    assert 'di' in data, 'variable not found in data: key=di'
    # initialize data
    Nc = data["Nc"]
    Ni = data["Ni"]
    xc = data["xc"]
    wc = data["wc"]
    dc = data["dc"]
    wi = data["wi"]
    di = data["di"]

def init_params(data, params):
    # initialize data
    Nc = data["Nc"]
    Ni = data["Ni"]
    xc = data["xc"]
    wc = data["wc"]
    dc = data["dc"]
    wi = data["wi"]
    di = data["di"]
    # assign init values for parameters
    params["phi"] = pyro.sample("phi", dist.Uniform(0., 1, dims=(2, 2)))
    params["q"] = pyro.sample("q", dist.Uniform(0., 1))
    params["beta0C"] = pyro.sample("beta0C"))
    params["beta"] = pyro.sample("beta"))

def model(data, params):
    # initialize data
    Nc = data["Nc"]
    Ni = data["Ni"]
    xc = data["xc"]
    wc = data["wc"]
    dc = data["dc"]
    wi = data["wi"]
    di = data["di"]
    
    # init parameters
    phi = params["phi"]
    q = params["q"]
    beta0C = params["beta0C"]
    beta = params["beta"]
    # initialize transformed parameters
    # model block

    for n in range(1, to_int(Nc) + 1):

        xc[n - 1] =  _pyro_sample(_index_select(xc, n - 1) , "xc[%d]" % (to_int(n-1)), "bernoulli", [q], obs=_index_select(xc, n - 1) )
        dc[n - 1] =  _pyro_sample(_index_select(dc, n - 1) , "dc[%d]" % (to_int(n-1)), "bernoulli_logit", [(beta0C + (beta * _index_select(xc, n - 1) ))], obs=_index_select(dc, n - 1) )
        wc[n - 1] =  _pyro_sample(_index_select(wc, n - 1) , "wc[%d]" % (to_int(n-1)), "bernoulli", [_index_select(_index_select(phi, (xc[n - 1] + 1) - 1) , (dc[n - 1] + 1) - 1) ], obs=_index_select(wc, n - 1) )
    for n in range(1, to_int(Ni) + 1):

        di[n - 1] =  _pyro_sample(_index_select(di, n - 1) , "di[%d]" % (to_int(n-1)), "bernoulli", [((_call_func("inv_logit", [(beta0C + beta)]) * q) + (_call_func("inv_logit", [beta0C]) * (1 - q)))], obs=_index_select(di, n - 1) )
        wi[n - 1] =  _pyro_sample(_index_select(wi, n - 1) , "wi[%d]" % (to_int(n-1)), "bernoulli", [((_index_select(_index_select(phi, 1 - 1) , (di[n - 1] + 1) - 1)  * (1 - q)) + (_index_select(_index_select(phi, 2 - 1) , (di[n - 1] + 1) - 1)  * q))], obs=_index_select(wi, n - 1) )
    q =  _pyro_sample(q, "q", "uniform", [0., 1])
    beta0C =  _pyro_sample(beta0C, "beta0C", "normal", [0., 320])
    beta =  _pyro_sample(beta, "beta", "normal", [0., 320])
    for i in range(1, 2 + 1):
        for j in range(1, 2 + 1):
            phi[i - 1][j - 1] =  _pyro_sample(_index_select(_index_select(phi, i - 1) , j - 1) , "phi[%d][%d]" % (to_int(i-1),to_int(j-1)), "uniform", [0., 1])

