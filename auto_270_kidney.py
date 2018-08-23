# model file: ../example-models/bugs_examples/vol1/kidney/kidney.stan
import torch
import pyro
from pyro_utils import (to_float, _pyro_sample, _call_func, check_constraints,
init_real, init_vector, init_simplex, init_matrix, init_int, _index_select, to_int, _pyro_assign, as_bool)
def validate_data_def(data):
    assert 'NP' in data, 'variable not found in data: key=NP'
    assert 'N_uc' in data, 'variable not found in data: key=N_uc'
    assert 'N_rc' in data, 'variable not found in data: key=N_rc'
    assert 't_uc' in data, 'variable not found in data: key=t_uc'
    assert 't_rc' in data, 'variable not found in data: key=t_rc'
    assert 'disease_uc' in data, 'variable not found in data: key=disease_uc'
    assert 'disease_rc' in data, 'variable not found in data: key=disease_rc'
    assert 'patient_uc' in data, 'variable not found in data: key=patient_uc'
    assert 'patient_rc' in data, 'variable not found in data: key=patient_rc'
    assert 'sex_uc' in data, 'variable not found in data: key=sex_uc'
    assert 'sex_rc' in data, 'variable not found in data: key=sex_rc'
    assert 'age_uc' in data, 'variable not found in data: key=age_uc'
    assert 'age_rc' in data, 'variable not found in data: key=age_rc'
    # initialize data
    NP = data["NP"]
    N_uc = data["N_uc"]
    N_rc = data["N_rc"]
    t_uc = data["t_uc"]
    t_rc = data["t_rc"]
    disease_uc = data["disease_uc"]
    disease_rc = data["disease_rc"]
    patient_uc = data["patient_uc"]
    patient_rc = data["patient_rc"]
    sex_uc = data["sex_uc"]
    sex_rc = data["sex_rc"]
    age_uc = data["age_uc"]
    age_rc = data["age_rc"]
    check_constraints(NP, low=0, dims=[1])
    check_constraints(N_uc, low=0, dims=[1])
    check_constraints(N_rc, low=0, dims=[1])
    check_constraints(t_uc, low=0, dims=[N_uc])
    check_constraints(t_rc, low=0, dims=[N_rc])
    check_constraints(disease_uc, dims=[N_uc])
    check_constraints(disease_rc, dims=[N_rc])
    check_constraints(patient_uc, dims=[N_uc])
    check_constraints(patient_rc, dims=[N_rc])
    check_constraints(sex_uc, dims=[N_uc])
    check_constraints(sex_rc, dims=[N_rc])
    check_constraints(age_uc, dims=[N_uc])
    check_constraints(age_rc, dims=[N_rc])

def init_params(data, params):
    # initialize data
    NP = data["NP"]
    N_uc = data["N_uc"]
    N_rc = data["N_rc"]
    t_uc = data["t_uc"]
    t_rc = data["t_rc"]
    disease_uc = data["disease_uc"]
    disease_rc = data["disease_rc"]
    patient_uc = data["patient_uc"]
    patient_rc = data["patient_rc"]
    sex_uc = data["sex_uc"]
    sex_rc = data["sex_rc"]
    age_uc = data["age_uc"]
    age_rc = data["age_rc"]
    # assign init values for parameters
    params["alpha"] = init_real("alpha") # real/double
    params["beta_age"] = init_real("beta_age") # real/double
    params["beta_sex"] = init_real("beta_sex") # real/double
    params["beta_disease2"] = init_real("beta_disease2") # real/double
    params["beta_disease3"] = init_real("beta_disease3") # real/double
    params["beta_disease4"] = init_real("beta_disease4") # real/double
    params["r"] = init_real("r", low=0) # real/double
    params["tau"] = init_real("tau", low=0) # real/double
    params["b"] = init_real("b", dims=(NP)) # real/double

def model(data, params):
    # initialize data
    NP = data["NP"]
    N_uc = data["N_uc"]
    N_rc = data["N_rc"]
    t_uc = data["t_uc"]
    t_rc = data["t_rc"]
    disease_uc = data["disease_uc"]
    disease_rc = data["disease_rc"]
    patient_uc = data["patient_uc"]
    patient_rc = data["patient_rc"]
    sex_uc = data["sex_uc"]
    sex_rc = data["sex_rc"]
    age_uc = data["age_uc"]
    age_rc = data["age_rc"]
    # INIT parameters
    alpha = params["alpha"]
    beta_age = params["beta_age"]
    beta_sex = params["beta_sex"]
    beta_disease2 = params["beta_disease2"]
    beta_disease3 = params["beta_disease3"]
    beta_disease4 = params["beta_disease4"]
    r = params["r"]
    tau = params["tau"]
    b = params["b"]
    # initialize transformed parameters
    sigma = init_real("sigma") # real/double
    yabeta_disease = init_real("yabeta_disease", dims=(4)) # real/double
    yabeta_disease[1 - 1] = _pyro_assign(yabeta_disease[1 - 1], 0)
    yabeta_disease[2 - 1] = _pyro_assign(yabeta_disease[2 - 1], beta_disease2)
    yabeta_disease[3 - 1] = _pyro_assign(yabeta_disease[3 - 1], beta_disease3)
    yabeta_disease[4 - 1] = _pyro_assign(yabeta_disease[4 - 1], beta_disease4)
    sigma = _pyro_assign(sigma, _call_func("sqrt", [(1 / tau)]))
    # model block

    alpha =  _pyro_sample(alpha, "alpha", "normal", [0, 100])
    beta_age =  _pyro_sample(beta_age, "beta_age", "normal", [0, 100])
    beta_sex =  _pyro_sample(beta_sex, "beta_sex", "normal", [0, 100])
    beta_disease2 =  _pyro_sample(beta_disease2, "beta_disease2", "normal", [0, 100])
    beta_disease3 =  _pyro_sample(beta_disease3, "beta_disease3", "normal", [0, 100])
    beta_disease4 =  _pyro_sample(beta_disease4, "beta_disease4", "normal", [0, 100])
    tau =  _pyro_sample(tau, "tau", "gamma", [0.001, 0.001])
    r =  _pyro_sample(r, "r", "gamma", [1, 0.001])
    for i in range(1, to_int(NP) + 1):
        b[i - 1] =  _pyro_sample(_index_select(b, i - 1) , "b[%d]" % (to_int(i-1)), "normal", [0, sigma])
    for i in range(1, to_int(N_uc) + 1):

        t_uc[i - 1] =  _pyro_sample(_index_select(t_uc, i - 1) , "t_uc[%d]" % (to_int(i-1)), "weibull", [r, _call_func("exp", [(-(((((alpha + (beta_age * _index_select(age_uc, i - 1) )) + (beta_sex * _index_select(sex_uc, i - 1) )) + _index_select(yabeta_disease, disease_uc[i - 1] - 1) ) + _index_select(b, patient_uc[i - 1] - 1) )) / r)])], obs=_index_select(t_uc, i - 1) )
    for i in range(1, to_int(N_rc) + 1):

        
