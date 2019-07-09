from __future__ import absolute_import, division, print_function

import math
import warnings

import torch

from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan


class ESS(ELBO):
    def __init__(self,
                 num_inner=1000,
                 num_outer=10,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=True,
                 strict_enumeration_warning=True):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting
        self.num_inner = num_inner
        self.num_outer = num_outer

        super(ESS, self).__init__(num_particles=num_inner*num_outer,
                                        max_plate_nesting=max_plate_nesting,
                                        vectorize_particles=vectorize_particles,
                                        strict_enumeration_warning=strict_enumeration_warning)

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, *args, **kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0.

            # compute elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    if is_vectorized:
                        log_prob_sum = site["log_prob"].detach().reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = torch_item(site["log_prob_sum"])

                    elbo_particle = elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    if is_vectorized:
                        log_prob_sum = log_prob.detach().reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = torch_item(site["log_prob_sum"])

                    elbo_particle = elbo_particle - log_prob_sum

            elbo_particles.append(elbo_particle)

        if is_vectorized:
            elbo_particles = elbo_particles[0]
        else:
            elbo_particles = torch.tensor(elbo_particles)  # no need to use .new*() here

        #print('elbo_particles', elbo_particles.size())
        elbo_particles = elbo_particles.view(self.num_outer, self.num_inner)
        #sys.exit()

        log_w_norm = elbo_particles - torch.logsumexp(elbo_particles, dim=1, keepdim=True)
        ess_val = torch.exp(-torch.logsumexp(2*log_w_norm, dim=1))

        #print('ess_val', ess_val.size())
        #sys.exit()

        loss = ess_val.mean()
        warn_if_nan(loss, "loss")
        return loss.item()
