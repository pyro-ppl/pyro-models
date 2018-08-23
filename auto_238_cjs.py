# model file: ../example-models/misc/ecology/mark-recapture/cjs.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'history' in data, 'variable not found in data: key=history'
    # initialize data
    history = data["history"]
    check_constraints(history, low=0, dims=[7])

def init_params(data, params):
    # initialize data
    history = data["history"]
    # assign init values for parameters
    params["phi"] = init_real("phi", low=0, high=1, dims=(2)) # real/double
    params["p"] = init_real("p", low=0, high=1, dims=(3)) # real/double

def model(data, params):
    # initialize data
    history = data["history"]
    # INIT parameters
    phi = params["phi"]
    p = params["p"]
    # initialize transformed parameters
    chi = init_real("chi", low=0, high=1, dims=(3)) # real/double
    chi[3 - 1] = _pyro_assign(chi[3 - 1], 1)
    chi[2 - 1] = _pyro_assign(chi[2 - 1], ((1 - _index_select(phi, 2 - 1) ) + (_index_select(phi, 2 - 1)  * (1 - _index_select(p, 3 - 1) ))))
    chi[1 - 1] = _pyro_assign(chi[1 - 1], ((1 - _index_select(phi, 1 - 1) ) + ((_index_select(phi, 1 - 1)  * (1 - _index_select(p, 2 - 1) )) * _index_select(chi, 2 - 1) )))
    # model block

    pyro.sample("(_index_select(history, 7 - 1)  * (((_call_func( log , [_index_select(phi, 1 - 1) ]) + _call_func( log , [_index_select(p, 2 - 1) ])) + _call_func( log , [_index_select(phi, 2 - 1) ])) + _call_func( log , [_index_select(p, 3 - 1) ])))", dist.Bernoulli((_index_select(history, 7 - 1)  * (((_call_func("log", [_index_select(phi, 1 - 1) ]) + _call_func("log", [_index_select(p, 2 - 1) ])) + _call_func("log", [_index_select(phi, 2 - 1) ])) + _call_func("log", [_index_select(p, 3 - 1) ])))), obs=(1));
    pyro.sample("(_index_select(history, 6 - 1)  * ((_call_func( log , [_index_select(phi, 1 - 1) ]) + _call_func( log , [_index_select(p, 2 - 1) ])) + _call_func( log , [_index_select(chi, 2 - 1) ])))", dist.Bernoulli((_index_select(history, 6 - 1)  * ((_call_func("log", [_index_select(phi, 1 - 1) ]) + _call_func("log", [_index_select(p, 2 - 1) ])) + _call_func("log", [_index_select(chi, 2 - 1) ])))), obs=(1));
    pyro.sample("(_index_select(history, 5 - 1)  * (((_call_func( log , [_index_select(phi, 1 - 1) ]) + _call_func( log1m , [_index_select(p, 2 - 1) ])) + _call_func( log , [_index_select(phi, 2 - 1) ])) + _call_func( log , [_index_select(p, 3 - 1) ])))", dist.Bernoulli((_index_select(history, 5 - 1)  * (((_call_func("log", [_index_select(phi, 1 - 1) ]) + _call_func("log1m", [_index_select(p, 2 - 1) ])) + _call_func("log", [_index_select(phi, 2 - 1) ])) + _call_func("log", [_index_select(p, 3 - 1) ])))), obs=(1));
    pyro.sample("(_index_select(history, 4 - 1)  * _call_func( log , [_index_select(chi, 1 - 1) ]))", dist.Bernoulli((_index_select(history, 4 - 1)  * _call_func("log", [_index_select(chi, 1 - 1) ]))), obs=(1));
    pyro.sample("(_index_select(history, 3 - 1)  * (_call_func( log , [_index_select(phi, 2 - 1) ]) + _call_func( log , [_index_select(p, 3 - 1) ])))", dist.Bernoulli((_index_select(history, 3 - 1)  * (_call_func("log", [_index_select(phi, 2 - 1) ]) + _call_func("log", [_index_select(p, 3 - 1) ])))), obs=(1));
    pyro.sample("(_index_select(history, 2 - 1)  * _call_func( log , [_index_select(chi, 2 - 1) ]))", dist.Bernoulli((_index_select(history, 2 - 1)  * _call_func("log", [_index_select(chi, 2 - 1) ]))), obs=(1));

