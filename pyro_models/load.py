import pyro_models
import os
import importlib
import imp
from functools import partial

def load():
    model_dir = pyro_models.__path__[0]
    model_files = [ f for f in os.listdir(model_dir)  if os.path.isfile(os.path.join(model_dir, f)) and f[0].isdigit() ]
    models = {}

    def model_wrapped(foo, data, params):
        # we need to wrap init_params in the model because variables declared
        # in Stan's "parameters" block are actually random variables
        foo.model(data, foo.init_params(data))

    for f in model_files:
        number, filename = f.split('_', 1)
        model = models.setdefault(int(number), {})

        if filename.endswith('.py'):
            model['model_file'] = os.path.join(model_dir, f)
            model['model_name'] = filename[:-3].replace('_', ' ').lower()
            model['index'] = number
            foo = imp.load_source('model.'+filename[:-3].lower(), os.path.join(model_dir, f))
            model['module'] = foo 

            # NOTE: Need to use partial to make sure correct foo is bound to function!
            model['model'] = partial(model_wrapped, foo)

        elif filename.endswith('.py.json'):
            model['data_file'] = os.path.join(model_dir, f)
        else:
            raise Exception(f'Invalid file {f} in model zoo')

    # TODO: Check that there is a data file for every model file

    return models