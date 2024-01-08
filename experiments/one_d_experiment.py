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

from config import default_one_d_exp_parameters

from ise.acquisitions.discrete_safe_acquisition_optimizer import DiscreteSafeAcquisitionOptimizer
from ise.acquisitions.safe_opt_acquisition import SafeOptAcquisition
from ise.acquisitions.info_theoretic_safe_bo_acquition import InfoTheoreticSafeBoAcquisition
from ise.acquisitions.mes_wrapper_acquisition import MesAcquisitionWrapper

import experiments.experiment_helpers as helpers

def test_function(x):
    exp_1 = torch.exp(-torch.pow(x, 1)) + 0.005
    exp_2 = torch.exp(-torch.pow(x - 4, 2)) + 0.005
    exp_3 = -torch.exp(-torch.pow(x - 7, 2))
    exp_4 = torch.exp(-torch.pow(x - 10, 2))

    return (exp_1 + 15 * exp_2 + 3 * exp_3 + 18 * exp_4).squeeze(0) + 0.33


def run_experiment(parameters):
    beta = parameters["beta"]
    domain = parameters["domain"]
    random_seed = parameters["random_seed"]
    observation_noise_variance = parameters["noise_variance"]
    kernel_lengthscale = parameters["kernel_lengthscale"]
    method = parameters["method"]
    gp_prior_mean = 0
    output_scale = 50

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    gp_model = helpers.get_gp_model(
        gp_prior_mean, kernel_lengthscale, output_scale, observation_noise_variance, beta)
    safe_seed = torch.zeros(1, len(domain))
    safe_seed_value = test_function(safe_seed)
    gp_model.add_observations(safe_seed, safe_seed_value)
    while gp_model.lower_confidence_bound(safe_seed) < 0:
        gp_model.add_observations(safe_seed, safe_seed_value)

    sample_points_number = 1000
    if method == 'safe_opt':
        lipschitz_constant = parameters["lipschitz_constant"]
        acquisition = SafeOptAcquisition(gp_model, gp_model, safe_seed, domain, lipschitz_constant, sample_points_number)
    elif method == 'ise_mes':
        learning_rate = parameters["learning_rate"]
        epochs = parameters["epochs"]
        acquisition = InfoTheoreticSafeBoAcquisition(
            gp_model, gp_model, domain, safe_seed, sample_points_number, learning_rate, epochs, 50)
    elif method == 'mes_safe':
        candidate_set = torch.rand(50, len(domain), dtype=torch.tensor(domain).dtype)
        candidate_set = torch.tensor(domain)[0, 0] + \
                        (torch.tensor(domain)[0, 1] - torch.tensor(domain)[0, 0]) * candidate_set
        pure_mes_acquisition = qMaxValueEntropy(gp_model, candidate_set)
        acquisition = DiscreteSafeAcquisitionOptimizer(
            gp_model, safe_seed, pure_mes_acquisition, domain, sample_points_number)
    elif method == 'mes':
        acquisition = MesAcquisitionWrapper(gp_model, domain)
    else:
        raise ValueError(f'Requested method "{method}" not available!')

    number_of_iterations = 300
    max_value = test_function(torch.tensor(4))
    simple_regret = max_value - safe_seed_value
    for i in range(number_of_iterations):
        next_point_to_sample, acquisition_value = acquisition.optimize()
        noisy_observation = test_function(next_point_to_sample) + torch.randn(1) * observation_noise_variance
        gp_model.add_observations(next_point_to_sample, noisy_observation)

        instantaneous_regret = max_value - test_function(next_point_to_sample)
        simple_regret = instantaneous_regret if instantaneous_regret < simple_regret else simple_regret

        # log artifacts


if __name__ == "__main__":
    run_experiment(default_one_d_exp_parameters)
