import json
import os
from six import string_types

import torch

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
            raise ValueError("Invalid tensorization of data dict")
    for k in to_delete:
        del data[k]

def data(model):
    data = json_file_to_mem_format(model['data_file'])
    foo = model['module']

    # convert dicts to torch tensors
    tensorize_data(data)
    model['data'] = data

    if 'transformed_data' in dir(foo):
        # run transformed_data block if it exists
        foo.transformed_data(data)

    return data