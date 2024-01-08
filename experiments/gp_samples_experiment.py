# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Alessandro G. Bottero, AlessandroGiacomo.Bottero@de.bosch.com

import numpy as np
import torch

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

from config import default_gp_samples_exp_parameters

from ise.acquisitions.discrete_safe_acquisition_optimizer import DiscreteSafeAcquisitionOptimizer
from ise.acquisitions.safe_opt_acquisition import SafeOptAcquisition
from ise.acquisitions.info_theoretic_safe_bo_acquition import InfoTheoreticSafeBoAcquisition
from ise.acquisitions.mes_wrapper_acquisition import MesAcquisitionWrapper
from ise.utils.generic_utils import get_gp_sample_safe_at_origin, get_gp_sample_bounded_at_origin

import experiments.experiment_helpers as helpers


def run_regret_experiment(parameters):
    beta = parameters["beta"]
    domain = parameters["domain"]
    random_seed = parameters["random_seed"]
    observation_noise_variance = parameters["noise_variance"]
    kernel_lengthscale = parameters["kernel_lengthscale"]
    output_scale = parameters["output_scale"]
    method = parameters["method"]
    mes_candidate_set_size = parameters["mes_candidate_set_size"]
    objective_is_constraint = parameters["objective_is_constraint"]
    gp_prior_mean = 0

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    gp_model_objective = helpers.get_gp_model(
        gp_prior_mean, kernel_lengthscale, output_scale, observation_noise_variance, beta)
    gp_model_constraint = helpers.get_gp_model(
        gp_prior_mean, kernel_lengthscale, output_scale, observation_noise_variance, beta)
    constraint = get_gp_sample_safe_at_origin(gp_model_constraint, domain, observation_noise_variance)
    if objective_is_constraint:
        objective = constraint
    else:
        objective = get_gp_sample_bounded_at_origin(gp_model_objective, domain, 0.5, observation_noise_variance)
    safe_seed = torch.zeros(1, len(domain))
    safe_seed_value_objective = objective(safe_seed)
    safe_seed_value_constraint = constraint(safe_seed)
    gp_model_objective.add_observations(safe_seed, safe_seed_value_objective)
    gp_model_constraint.add_observations(safe_seed, safe_seed_value_constraint)
    while gp_model_constraint.lower_confidence_bound(safe_seed) < 0:
        gp_model_constraint.add_observations(safe_seed, safe_seed_value_constraint)

    number_of_samples_per_dimension = 200
    number_of_samples = np.power(number_of_samples_per_dimension, len(domain))
    if method == 'safe_opt':
        lipschitz_constant = parameters["lipschitz_constant"]
        acquisition = SafeOptAcquisition(
            gp_model_constraint, gp_model_objective, safe_seed, domain, lipschitz_constant, number_of_samples)
    elif method == 'ise_mes':
        learning_rate = parameters["learning_rate"]
        epochs = parameters["epochs"]
        acquisition = InfoTheoreticSafeBoAcquisition(
            gp_model_constraint,
            gp_model_objective,
            domain,
            safe_seed,
            number_of_samples,
            learning_rate,
            epochs,
            mes_candidate_set_size)
    elif method == 'mes_safe':
        candidate_set = torch.rand(50, len(domain), dtype=torch.tensor(domain).dtype)
        transposed_domain = torch.tensor(domain).T
        candidate_set = transposed_domain[0] + (transposed_domain[1] - transposed_domain[0]) * candidate_set
        pure_mes_acquisition = qMaxValueEntropy(gp_model_objective, candidate_set)
        acquisition = DiscreteSafeAcquisitionOptimizer(
            gp_model_objective, safe_seed, pure_mes_acquisition, domain, number_of_samples)
    elif method == 'mes':
        acquisition = MesAcquisitionWrapper(gp_model_objective, domain)
    else:
        raise ValueError(f'Requested method "{method}" not available!')


    number_of_iterations = 130
    max_value = safe_seed_value_objective
    for i in range(number_of_iterations):
        next_point_to_sample, acquisition_value = acquisition.optimize()
        noisy_observation_constraint = constraint(next_point_to_sample, True)
        if objective_is_constraint:
            noisy_observation_objective = noisy_observation_constraint
        else:
            noisy_observation_objective = objective(next_point_to_sample, True)
        gp_model_objective.add_observations(next_point_to_sample, noisy_observation_objective)
        gp_model_constraint.add_observations(next_point_to_sample, noisy_observation_constraint)

        objective_value = objective(next_point_to_sample)
        max_value = max_value if objective_value < max_value else objective_value

        safety_observation = constraint(next_point_to_sample, False)

        # log metrics


if __name__ == "__main__":
    run_regret_experiment(default_gp_samples_exp_parameters)
