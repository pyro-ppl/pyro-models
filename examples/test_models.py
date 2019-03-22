import argparse
import sys

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal
import pyro.optim as optim

import pyro_models

def select_model(args, models):
    # Check that model is specified and exists
    if args.model_name is not None and args.model_name not in models:
        raise Exception(f'Model named {args.model_name} is not present in model zoo!')    
    elif args.model_name is None:
        raise Exception(f'Model name not specified in command arguments!')

    return models[args.model_name]

def main(args):
    # Init Pyro
    pyro.enable_validation(True)
    pyro.clear_param_store()

    # Load meta-data for all models and select model based on command arguments
    models = pyro_models.load()
    model_dict = select_model(args, models)
    #model_dict = models['arm.radon_inter_vary']
    #model_dict = models['arm.radon_group']
    #model_dict = models['arm.wells_dist']
    #model_dict = models['arm.earnings_latin_square_chr']
    #model_dict = models['arm.anova_radon_nopred']

    # Define model/data/guide
    model = model_dict['model']
    data = pyro_models.data(model_dict)
    #guide = AutoDiagonalNormal(model)
    guide = AutoMultivariateNormal(model)
    #guide = AutoDelta(model)

    # Perform variational inference
    svi = SVI(model, guide, optim.Adam({'lr': 0.1}), loss=Trace_ELBO(vectorize_particles=True, num_particles=10))
    iwae = RenyiELBO(vectorize_particles=True, num_particles=5000)

    for i in range(args.num_epochs):
        params = {}
        loss = svi.step(data, params)
        print('elbo', loss,'| iwae', iwae.loss(model, guide, data, {}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help="number of epochs to run learning for")
    parser.add_argument('-m', '--model-name', type=str, help="model name qualified by dataset")
    args = parser.parse_args()
    main(args)
