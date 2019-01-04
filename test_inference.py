from utils import (EPSILON, get_fns_pyro, tensorize_data,
                   json_file_to_mem_format, log_traceback,
                   import_by_string, exists_p, load_p, save_p)
import copy
import sys
import pystan
import numpy as np
import contextlib
import functools
import argparse

import pyro.poutine as poutine
from pdb import set_trace as bb
import pyro
import torch
import json
import os.path
from torch.distributions import constraints
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.mcmc import MCMC
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal
import pyro.optim as optim
from run_compiler_all_examples import get_all_data_paths, get_cached_state


class suppress_pystan_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C sub-function.
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def _product(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def run_stan_nuts(data, sfile, n_samples=200, model_cache=None, suppress=True):
    if model_cache is not None and exists_p(model_cache):
        sm = load_p(model_cache)
    else:
        with open(sfile, "r") as f:
                code = f.read()
        sm = pystan.StanModel(model_code=code)
        if model_cache is not None:
            save_p(sm, model_cache)
    suppress = False
    context = suppress_pystan_stderr() if suppress else contextlib.suppress()
    with context:
#         fit = sm.sampling(data=data, init=0, seed=100, iter=n_samples, algorithm="NUTS")
        fit = sm.sampling(data=data, seed=100, iter=n_samples, algorithm="NUTS")
        site_values=fit.extract(permuted=True)
        site_keys = site_values.keys()
#         print(fit)
        return {k: (np.mean(site_values[k], axis=0), np.std(site_values[k], axis=0)) for k in site_keys}


def run_pyro_nuts(data, pfile, n_samples, params):

    # import model, transformed_data functions (if exists) from pyro module

    model = import_by_string(pfile + ".model")
    assert model is not None, "model couldn't be imported"
    transformed_data = import_by_string(pfile + ".transformed_data")
    if transformed_data is not None:
        transformed_data(data)

    nuts_kernel = NUTS(model, step_size=0.0855)
    mcmc_run = MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=int(n_samples/2))
    posteriors = {k: [] for k in params}

    for trace, _ in mcmc_run._traces(data, params):
        for k in posteriors:
            posteriors[k].append(trace.nodes[k]['value'])


    #posteriors["sigma"] = list(map(torch.exp, posteriors["log_sigma"]))
    #del posteriors["log_sigma"]

    posterior_means = {k: torch.mean(torch.stack(posteriors[k]), 0) for k in posteriors}
    return posterior_means


def run_pyro_advi(data, validate_data_def, initialized_params, model, num_epochs=100, num_runs=1, guide='none', inits=None):
    # fix initializations
    latent_dims = sum(_product(tens.shape) for tens in initialized_params.values())
    if inits:
        pyro.param('auto_loc', inits[0], constraint=constraints.real)
        pyro.param('auto_scale', inits[1], constraint=constraints.positive)
#     else:
#         pyro.param('auto_loc', torch.zeros(latent_dims), constraint=constraints.real)
#         pyro.param('auto_scale', torch.zeros(latent_dims) + 0.01, constraint=constraints.positive)
    if guide == 'diagnormal':
        guide = AutoDiagonalNormal(model)
    elif guide == 'mvn':
        guide = AutoMultivariateNormal(model)
    else:
        guide = AutoDelta(model)
    adam = optim.SGD({'lr': 1e-3})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    total_d_elbos = 0
    for i in range(num_runs):
        pyro.clear_param_store()
        for i in range(num_epochs):
            loss = svi.step(data, initialized_params)
            if i == 0:
                initial_elbo = loss
            if i == num_epochs - 1:
                total_d_elbos += loss - initial_elbo
                print('loss=', loss)
    mean_elbo = total_d_elbos / num_runs
    return pyro.param('auto_loc'), pyro.param('auto_scale'), mean_elbo


def compare_models(data, sfile, pfile, n_samples=200, num_epochs=100, num_runs=1, guide='none', model_cache=None, status={}, metrics=[]):
    copy_data = copy.deepcopy(data)
    validate_data_def, init_params, model, transformed_data = get_fns_pyro(pfile)

    initialized_params = {}
    tensorize_data(data)
    if transformed_data is not None:
        transformed_data(data)
    init_params(data, initialized_params)
    try:
#         stan_metrics = run_stan_nuts(copy_data, sfile, n_samples=n_samples, model_cache=model_cache)
#         all_locs = np.append(stan_metrics['eta'][0], [stan_metrics['mu'][0], stan_metrics['sigma_eta'][0], stan_metrics['sigma_y'][0]])
#         all_locs = torch.tensor(all_locs).float()
#         all_scales = np.append(stan_metrics['eta'][1], [stan_metrics['mu'][1], stan_metrics['sigma_eta'][1], stan_metrics['sigma_y'][1]])
#         all_scales = torch.tensor(all_scales).float()
        pyro_metrics = run_pyro_advi(data, validate_data_def, initialized_params, model, num_epochs,
                                     num_runs, guide=guide)  #, inits=[all_locs.clone(), all_scales.clone()])
        stan_val = 1
        if np.abs(stan_val - pyro_metrics[0].sum().item() / stan_val) < 0.05:
            status['success'] += 1
        else:
            status['failure'] += 1
        md = {'example': pfile, 'd_elbo': pyro_metrics[2]}
        metrics.append(md)
    except RuntimeError as e:
        if 'no latent variables' in str(e):
            status['no_latent'] += 1
        elif 'buffers have already been freed' in str(e):
            # backprop-ing twice
            status['grad'].append(pfile)
        elif 'modified by an inplace operation' in str(e):
            # backprop-ing twice
            status['grad_killed'].append(pfile)
        else: raise e
    print(status)
    print(metrics)
    return status, metrics


def use_log_prob_tested_models(args):
    cfname = "%s/status.pkl" % p_args.output_folder
    (j, status) = get_cached_state(cfname)
    successful_models = status[0]
    advi_status = {'success': 0, 'failure': 0, 'no_latent': 0, 'grad': [], 'grad_killed': []}
    metrics = [] # list of dicts
    # args = list(sorted(get_all_data_paths(p_args.examples_folder,ofldr)))
    print("%d =total possible (R data, stan model) pairs that pass log-prob test" % len(successful_models))

    eid = args.i if args.i else ''
    for dfile, mfile, pfile, model_cache, _ in successful_models:
        if os.path.exists(pfile):
            jfile = "%s.json" % pfile
            assert os.path.exists(jfile)
            with open(jfile, "r") as fj:
                file_data = json.load(fj)
            data = json_file_to_mem_format(file_data)
            if eid != '':
                if ("auto_%s_" % eid) in pfile:
                    print("PROCESSING:\nstan-file: %s\npyro-file: %s" % (mfile, pfile))
                    compare_models(data, mfile, pfile, model_cache=model_cache, n_samples=args.num_samples,
                                   num_epochs=args.num_epochs, num_runs=args.num_runs, guide=args.guide, status=advi_status,
                                   metrics=metrics)
                    return
            else:
                print("PROCESSING:\nstan-file: %s\npyro-file: %s" % (mfile, pfile))
                advi_status, metrics = compare_models(data, mfile, pfile, model_cache=model_cache, n_samples=args.num_samples,
                                                      num_epochs=args.num_epochs, num_runs=args.num_runs,
                                                      status=advi_status, metrics=metrics)
                print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-folder', default='./test_compiler', type=str,
                        help="Output folder from run_compiler_all_examples.py, should have a status.pkl file in it")
    parser.add_argument('-ns', '--num-samples', default=600, type=int,
                        help="num samples to draw for NUTS")
    parser.add_argument('-i', type=int, help="id of stan example (int)")
    parser.add_argument('-n', '--num-epochs', default=300, type=int,
                        help="num epochs to run SVI")
    parser.add_argument('-nr', '--num-runs', default=1, type=int,
                        help="num times to run to aggregate avg elbo")
    parser.add_argument('-g', '--guide', default='', type=str,
                        help="guide type: diagnormal, mvn, neural")
    p_args = parser.parse_args()
    use_log_prob_tested_models(p_args)

