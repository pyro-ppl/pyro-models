import pyro_models
import os
import importlib
import imp
from functools import partial

def load():
    base_dir = pyro_models.__path__[0]
    model_dirs = [ f for f in os.listdir(base_dir)  if not os.path.isfile(os.path.join(base_dir, f)) and not f.startswith('_') ]
    models = {}

    def model_wrapped(foo, data, params):
        # we need to wrap init_params in the model because variables declared
        # in Stan's "parameters" block are actually random variables
        foo.model(data, foo.init_params(data))

    for d in model_dirs:
      model_dir = os.path.join(base_dir, d)
      # Check that this is a python module
      if not os.path.isfile(os.path.join(model_dir, '__init__.py')):
        continue

      model_files = [ f for f in os.listdir(model_dir)  if os.path.isfile(os.path.join(model_dir, f)) and not f.startswith('_') ]

      for f in model_files:
          # Model name is qualified by source
          name, ext = f.split('.', 1)
          fullname = d + '.' + name.lower()
          model = models.setdefault(fullname, {})

          if f.endswith('.py'):
              model['dataset'] = d
              model['source_file'] = os.path.join(model_dir, f)
              model['name'] = name.lower()
              foo = imp.load_source('model.'+f[:-3].lower(), os.path.join(model_dir, f))
              model['module'] = foo 

              # NOTE: Need to use partial to make sure correct foo is bound to function!
              model['model'] = partial(model_wrapped, foo)

          elif f.endswith('.py.json'):
              model['data_file'] = os.path.join(model_dir, f)
          else:
              raise Exception(f'Invalid file {f} in model zoo')

    return models