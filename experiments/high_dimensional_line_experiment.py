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

from ise.acquisitions.info_theoretic_line_bo_acquisition import InfoTheoreticLineBoAcquisition
from ise.acquisitions.safe_opt_line_bo_acquisition import SafeOptLineBoAcquisition
from ise.acquisitions.mes_line_bo_acquisition import MesLineBoAcquisition
from config import default_parameters_line_bo
import experiment_helpers as helpers


def multi_optima_gaussian(x):
    dimensions = x.shape[1]
    translate_1 = torch.zeros((1, dimensions))
    translate_1[0, 0] = 2.7
    translate_2 = torch.zeros((1, dimensions))
    translate_2[0, 0] = 6
    exponent_0 = -torch.pow(torch.norm(x, dim=-1), 2)
    exponent_1 = -torch.pow(torch.norm(x - translate_1, dim=-1), 2)
    exponent_2 = -torch.pow(torch.norm(x - translate_2, dim=-1), 2)
    exponent_3 = -torch.pow(torch.norm(x + translate_1, dim=-1), 2)
    exponent_4 = -torch.pow(torch.norm(x + translate_2, dim=-1), 2)

    return 0.5 * torch.exp(exponent_0) + \
           1 * torch.exp(exponent_1) + \
           3 * torch.exp(exponent_2) + \
           1 * torch.exp(exponent_3) + \
           3 * torch.exp(exponent_4) + 0.2


def run_line_bo_experiment(parameters):
    beta = parameters["beta"]
    random_seed = parameters["random_seed"]
    dimensions = parameters["dimensions"]
    scaled_domain = parameters["scaled_domain"]
    method = parameters["method"]
    high_noise = parameters["high_noise"]

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if scaled_domain:
        scaling_factor = np.sqrt(2. / dimensions)
    else:
        scaling_factor = 1
    domain = [(-3. * scaling_factor, 3. * scaling_factor) for _ in range(dimensions)]
    domain[0] = (-8., 8.)

    gp_prior_mean = 0
    kernel_lengthscale = 1.6
    output_scale = 1
    low_noise = 0.03

    def observation_noise(x):
        noise_variances = torch.ones((len(x), 1))
        index_true = x[..., 0] <= 0
        noise_variances[index_true] *= low_noise
        noise_variances[torch.logical_not(index_true)] *= high_noise

        return noise_variances

    gp_model_obj = helpers.get_gp_model(
        gp_prior_mean, kernel_lengthscale, output_scale, low_noise, beta)

    gp_model_safe = helpers.get_gp_model(
        gp_prior_mean, kernel_lengthscale, output_scale, observation_noise, beta)

    objective = multi_optima_gaussian
    constraint = multi_optima_gaussian
    safe_seed = torch.zeros(1, dimensions)

    mes_candidates_number = 50

    safe_seed_noise = observation_noise(safe_seed).item()
    for i in range(11):
        safe_seed_noisy_value_obj = objective(safe_seed) + torch.randn(1) * np.sqrt(safe_seed_noise)
        safe_seed_noisy_value_constraint = \
            constraint(safe_seed) + torch.randn(1) * np.sqrt(observation_noise(safe_seed).item())
        gp_model_obj.add_observations(safe_seed, safe_seed_noisy_value_obj)
        gp_model_safe.add_observations(safe_seed, safe_seed_noisy_value_constraint, torch.tensor([safe_seed_noise]))

    if method == 'safe_opt':
        lipschitz_constant = parameters["lipschitz_constant"]
        acquisition = SafeOptLineBoAcquisition(gp_model_safe, gp_model_obj, safe_seed, domain, lipschitz_constant)
    elif method == 'ise_mes':
        acquisition = InfoTheoreticLineBoAcquisition(
            gp_model_safe, gp_model_obj, safe_seed, domain, mes_candidates_number, observation_noise)
    elif method == 'mes_safe':
        acquisition = MesLineBoAcquisition(gp_model_obj, safe_seed, domain, mes_candidates_number, gp_model_safe)
    elif method == 'mes':
        acquisition = MesLineBoAcquisition(gp_model_obj, safe_seed, domain, mes_candidates_number)
    else:
        raise ValueError(f'Requested method "{method}" not available!')

    max_value = 3.2
    simple_regret = 1e6
    number_of_iterations = 600
    for i in range(number_of_iterations):
        next_point_to_sample, acquisition_value = acquisition.optimize()
        noise = observation_noise(next_point_to_sample)
        noisy_observation_objective = \
            objective(next_point_to_sample) + torch.randn(1) * np.sqrt(noise.item())
        noisy_observation_safety = \
            constraint(next_point_to_sample) + torch.randn(1) * np.sqrt(noise.item())
        gp_model_obj.add_observations(next_point_to_sample, noisy_observation_objective)
        gp_model_safe.add_observations(next_point_to_sample, noisy_observation_safety, torch.tensor([noise]))

        instantaneous_regret = max_value - objective(next_point_to_sample)
        simple_regret = instantaneous_regret if instantaneous_regret < simple_regret else simple_regret

        # log metrics

if __name__ == '__main__':
    run_line_bo_experiment(default_parameters_line_bo)
