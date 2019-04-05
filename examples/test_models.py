import argparse
import sys

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal, AutoFlowNormal, AutoIAFNormal #AutoAutoregressiveNormal
import pyro.optim as optim

import pyro_models
from ess import ESS

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
    #model_dict = select_model(args, models)
    #model_dict = models['arm.radon_inter_vary']
    #model_dict = models['arm.radon_complete_pool']
    #model_dict = models['arm.wells_dae_inter_c']
    #model_dict = models['arm.earnings_latin_square_chr']
    #model_dict = models['arm.anova_radon_nopred']
    model_dict = models['arm.wells_dist']

    # Define model/data/guide
    model = model_dict['model']
    data = pyro_models.data(model_dict)
    #guide = AutoDiagonalNormal(model)
    #guide = AutoMultivariateNormal(model)
    #guide = AutoDelta(model)
    #guide = AutoAutoregressiveNormal(model)
    guide = AutoIAFNormal(model)

    print(guide(data, {}))
    sys.exit()

    """hidden_dim = 16
    latent_dim = 2
    arn = pyro.nn.AutoRegressiveNN(latent_dim, [latent_dim*3+1], param_dims=[hidden_dim]*3)
    flow = dist.DeepSigmoidalFlow(arn)

    def guide(data, params):
        N = data["N"]
        switched = data["switched"]
        dist_ = data["dist"]

        pyro.module("naf", flow)
        flow_dist = dist.TransformedDistribution(dist.Normal(0., 1.).expand([latent_dim]), [flow])

        return pyro.sample("beta", flow_dist)

    flow_dist = dist.TransformedDistribution(dist.Normal(0., 1.).expand([latent_dim]), [flow])
    z = flow_dist.sample()
    print(flow._cache_size)
    print(z, flow_dist.log_prob(z))
    sys.exit()"""

    # Perform variational inference
    svi = SVI(model, guide, optim.Adam({'lr': 0.0001}), loss=Trace_ELBO(vectorize_particles=True, num_particles=100))
    iwae = RenyiELBO(vectorize_particles=True, num_particles=5000)
    ess = ESS(vectorize_particles=True, num_particles=5000)

    for i in range(args.num_epochs):
        params = {}
        loss = svi.step(data, params)
        #if i % 10 == 0:
        print(f'epoch {i}: elbo', loss,'| iwae', iwae.loss(model, guide, data, {}),'| ess', ess.loss(model, guide, data, {}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help="number of epochs to run learning for")
    parser.add_argument('-m', '--model-name', type=str, help="model name qualified by dataset")
    args = parser.parse_args()
    main(args)
