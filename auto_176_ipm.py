# model file: ../example-models/BPA/Ch.11/ipm.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'nyears' in data, 'variable not found in data: key=nyears'
    assert 'y' in data, 'variable not found in data: key=y'
    assert 'J' in data, 'variable not found in data: key=J'
    assert 'R' in data, 'variable not found in data: key=R'
    assert 'm' in data, 'variable not found in data: key=m'
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    m = data["m"]
    check_constraints(nyears, dims=[1])
    check_constraints(y, dims=[nyears])
    check_constraints(J, dims=[(nyears - 1)])
    check_constraints(R, dims=[(nyears - 1)])
    check_constraints(m, dims=[(2 * (nyears - 1)), nyears])

def transformed_data(data):
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    m = data["m"]
    ny_minus_1 = init_int("ny_minus_1") # real/double
    data["ny_minus_1"] = ny_minus_1

def init_params(data, params):
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    m = data["m"]
    # initialize transformed data
    ny_minus_1 = data["ny_minus_1"]
    # assign init values for parameters
    params["sigma_y"] = init_real("sigma_y", low=0) # real/double
    params["N1"] = init_vector("N1", low=0, dims=(nyears)) # vector
    params["Nad"] = init_vector("Nad", low=0, dims=(nyears)) # vector
    params["mean_sjuv"] = init_real("mean_sjuv", low=0, high=1) # real/double
    params["mean_sad"] = init_real("mean_sad", low=0, high=1) # real/double
    params["mean_p"] = init_real("mean_p", low=0, high=1) # real/double
    params["mean_fec"] = init_real("mean_fec", low=0) # real/double

def model(data, params):
    # initialize data
    nyears = data["nyears"]
    y = data["y"]
    J = data["J"]
    R = data["R"]
    m = data["m"]
    # initialize transformed data
    ny_minus_1 = data["ny_minus_1"]
    # INIT parameters
    sigma_y = params["sigma_y"]
    N1 = params["N1"]
    Nad = params["Nad"]
    mean_sjuv = params["mean_sjuv"]
    mean_sad = params["mean_sad"]
    mean_p = params["mean_p"]
    mean_fec = params["mean_fec"]
    # initialize transformed parameters
    sjuv = init_vector("sjuv", low=0, high=1, dims=(ny_minus_1)) # vector
    sad = init_vector("sad", low=0, high=1, dims=(ny_minus_1)) # vector
    p = init_vector("p", low=0, high=1, dims=(ny_minus_1)) # vector
    f = init_vector("f", low=0, dims=(ny_minus_1)) # vector
    Ntot = init_vector("Ntot", low=0, dims=(nyears)) # vector
    pr = init_simplex("pr", dims=((2 * ny_minus_1))) # real/double
    rho = init_vector("rho", low=0, dims=(ny_minus_1)) # vector
    for t in range(1, to_int(ny_minus_1) + 1):

        sjuv[t - 1] = _pyro_assign(sjuv[t - 1], mean_sjuv)
        sad[t - 1] = _pyro_assign(sad[t - 1], mean_sad)
        p[t - 1] = _pyro_assign(p[t - 1], mean_p)
        f[t - 1] = _pyro_assign(f[t - 1], mean_fec)
    Ntot = _pyro_assign(Ntot, _call_func("add", [Nad,N1]))
    pr = _pyro_assign(pr, _call_func("marray", [nyears,sjuv,sad,p, pstream__]))
    for t in range(1, to_int(ny_minus_1) + 1):
        rho[t - 1] = _pyro_assign(rho[t - 1], (_index_select(R, t - 1)  * _index_select(f, t - 1) ))
    # model block

    N1[1 - 1] =  _pyro_sample(_index_select(N1, 1 - 1) , "N1[%d]" % (to_int(1-1)), "normal", [100, 100])
    Nad[1 - 1] =  _pyro_sample(_index_select(Nad, 1 - 1) , "Nad[%d]" % (to_int(1-1)), "normal", [100, 100])
    for t in range(2, to_int(nyears) + 1):
        # {
        mean1 = init_real("mean1") # real/double

        mean1 = _pyro_assign(mean1, (((_index_select(f, (t - 1) - 1)  / 2) * _index_select(sjuv, (t - 1) - 1) ) * _index_select(Ntot, (t - 1) - 1) ))
        N1[t - 1] =  _pyro_sample(_index_select(N1, t - 1) , "N1[%d]" % (to_int(t-1)), "real_poisson", [mean1])
        Nad[t - 1] =  _pyro_sample(_index_select(Nad, t - 1) , "Nad[%d]" % (to_int(t-1)), "real_binomial", [_index_select(Ntot, (t - 1) - 1) , _index_select(sad, (t - 1) - 1) ])
        # }
    y =  _pyro_sample(y, "y", "normal", [Ntot, sigma_y], obs=y)
    for t in range(1, to_int((2 * ny_minus_1)) + 1):

        m[t - 1] =  _pyro_sample(_index_select(m, t - 1) , "m[%d]" % (to_int(t-1)), "multinomial", [_index_select(pr, t - 1) ], obs=_index_select(m, t - 1) )
    J =  _pyro_sample(J, "J", "poisson", [rho], obs=J)

