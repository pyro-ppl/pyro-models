import torch
import pyro
import pyro.distributions as dist
import json
import argparse
import importlib


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
    from six import string_types
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
    pyro.clear_param_store()
    from pyro.infer import SVI, Trace_ELBO
    from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal
    import pyro.optim as optim
    module = importlib.import_module(args.fname[:-3])
    model = module.model
    guide = AutoDelta(model)
    svi = SVI(model, guide, optim.Adam({'lr': 0.1}), loss=Trace_ELBO())
    data = json_file_to_mem_format(args.fname + '.json')
    params = module.init_params(data)
    tensorize_data(data)
    if hasattr(module, 'transformed_data'):
        module.transformed_data(data)
    for i in range(args.num_epochs):
        loss = svi.step(data, params)
        print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10, type=int)
    parser.add_argument('-f', '--fname', default='100_pilots_chr.py', type=str)
    args = parser.parse_args()
    main(args)
