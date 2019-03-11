import argparse
import sys

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal
import pyro.optim as optim

import pyro_models

def select_model(args, models):
<<<<<<< HEAD
    if args.model_name is None and args.model_index is None:
        raise Exception('One of model name or model index must be specified in command arguments!')
    
    if args.model_name is not None and args.model_index is not None:
        raise Exception('Cannot specify both model name and index in command arguments!')

    if args.model_name is not None:
        those_models = [m for idx, m in models.items() if m['model_name'] == args.model_name]
        if those_models == []:
            raise Exception(f'Model named {args.model_name} is not present in model zoo!')
        elif len(those_models) > 1:
            raise Exception(f'Model {args.model_name} is not uniquely named in model zoo!')
        print(those_models)
        model_dict = those_models[0]
    else:
        if args.model_index not in models:
            raise Exception(f'Model index {args.model_index} is not present in model zoo!')
        model_dict = models[args.model_index]
    return model_dict
=======
    # Check that model is specified and exists
    if args.model_name is not None and args.model_name not in models:
        raise Exception(f'Model named {args.model_name} is not present in model zoo!')    
    elif args.model_name is None:
        raise Exception(f'Model name not specified in command arguments!')

    return models[args.model_name]
>>>>>>> b291c7360fcb360a34c52a02c35e6dd397d36c29

def main(args):
    # Init Pyro
    pyro.enable_validation(True)
    pyro.clear_param_store()

    # Load meta-data for all models and select model based on command arguments
    models = pyro_models.load()
    model_dict = select_model(args, models)

    # Define model/data/guide
    model = model_dict['model']
    data = pyro_models.data(model_dict)
    guide = AutoDelta(model)

    # Perform variational inference
    svi = SVI(model, guide, optim.Adam({'lr': 0.1}), loss=Trace_ELBO())
    for i in range(args.num_epochs):
        params = {}
        loss = svi.step(data, params)
        print(loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
<<<<<<< HEAD
    parser.add_argument('-n', '--num-epochs', default=100, type=int)
    parser.add_argument('-m', '--model-name', type=str, help="model name given by filename")
    parser.add_argument('-i', '--model-index', type=int, help="index of model given by prefix")
=======
    parser.add_argument('-n', '--num-epochs', default=100, type=int, help="number of epochs to run learning for")
    parser.add_argument('-m', '--model-name', type=str, help="model name qualified by dataset")
>>>>>>> b291c7360fcb360a34c52a02c35e6dd397d36c29
    args = parser.parse_args()
    main(args)
