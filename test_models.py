import json
import argparse
import importlib
from six import string_types

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal
import pyro.optim as optim

def json_file_to_mem_format(fname):
    with open(fname, "r") as f:
        rdata = json.load(f)
    data = {}
    n = len(rdata[0])
    for i in range(n):
        key = rdata[0][i]
        if key != "args":
            data[key] = rdata[1][i]
    return data


def tensorize_data(data):
    """
    Convert python lists of data into pytorch tensors
    in-place operation

    :param data: Python dict of data with variable names as keys and values as values
    :type data: dict
    """
    to_delete = []
    for k in data:
        if isinstance(data[k], float) or isinstance(data[k], int):
            pass
        elif isinstance(data[k], string_types):
            to_delete.append(k)
        elif isinstance(data[k], list):
            for s in data[k]:
                if isinstance(s, string_types):
                    to_delete.append(k)
                    break
            else:
                data[k] = torch.tensor(data[k]).float()
        elif isinstance(data[k], torch.Tensor):
            data[k] = torch.tensor(data[k]).float()
        else:
            raise ValueError("invalid tensorization of data dict")
    for k in to_delete:
        print("Deleting k=%s string data in dict" % k)
        del data[k]


def main(args):
    pyro.enable_validation(True)
    pyro.clear_param_store()
    module = importlib.import_module(args.fname[:-3])
    model_block = module.model
    def model(data, params):
        # we need to wrap init_params in the model because of how
        # Stan's parameter blocks work
        params = module.init_params(data)
        model_block(data, params)
    # MAP estimates
    guide = AutoDelta(model)
    svi = SVI(model, guide, optim.Adam({'lr': 0.1}), loss=Trace_ELBO())
    data = json_file_to_mem_format(args.fname + '.json')
    tensorize_data(data)
    if hasattr(module, 'transformed_data'):
        module.transformed_data(data)
    for i in range(args.num_epochs):
        params = {}
        loss = svi.step(data, params)
        print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10, type=int)
    parser.add_argument('-f', '--fname', type=str, help="python file path")
    args = parser.parse_args()
    main(args)
