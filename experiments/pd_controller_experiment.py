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

from config import default_pd_controller_exp_parameters

from ise.acquisitions.discrete_safe_acquisition_optimizer import DiscreteSafeAcquisitionOptimizer
from ise.acquisitions.safe_opt_acquisition import SafeOptAcquisition
from ise.acquisitions.info_theoretic_safe_bo_acquition import InfoTheoreticSafeBoAcquisition
from ise.acquisitions.mes_wrapper_acquisition import MesAcquisitionWrapper

import experiments.experiment_helpers as helpers

from gym_brt.envs import QubeSwingupEnv
from gym_brt.control import flip_and_hold_policy

def evaluate_objective_and_constraint(kd_params, noise_variance=None):
    num_steps = 1000
    frequency = 250
    policy = flip_and_hold_policy
    kd_alpha = kd_params[0, 0].numpy()
    kd_theta = kd_params[0, 1].numpy()
    with QubeSwingupEnv(use_simulator=True, frequency=frequency, batch_size=num_steps) as env:
        state = env.reset_in_state(np.array([0, 0.1, 0.2, 0.], dtype=np.float64))
        alphas = []
        thetas = []
        alphas_dot = []
        thetas_dot = []
        actions = []
        step = 0
        done = False
        while not done and step <= num_steps:
            action, using_pd_controller = policy(state, kd_theta, kd_alpha)
            state, reward, done, info = env.step(action)
            thetas.append(np.abs(state[0]))
            alphas.append(np.abs(state[1]))
            alphas_dot.append(np.abs(state[2]))
            thetas_dot.append(np.abs(state[3]))
            actions.append(np.abs(action))
            step += 1

        objective = np.array(alphas_dot).sum() + np.array(thetas_dot).sum() + np.array(actions[10:]).sum()
        safety = np.array(alphas).sum()
        
        scaled_objective = torch.tensor(- objective / 100 + 6, dtype=torch.float32).unsqueeze(0)
        scaled_safety = torch.tensor(- safety / 3 + 6, dtype=torch.float32).unsqueeze(0)

        if noise_variance is not None:
            return scaled_objective + np.sqrt(noise_variance) * torch.rand(scaled_objective.shape), \
                scaled_safety + np.sqrt(noise_variance) * torch.rand(scaled_safety.shape)
        
        return scaled_objective, scaled_safety
            

def run_pd_controller_experiment(parameters):
    random_seed = parameters["random_seed"]
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    gp_prior_mean = 0
    kernel_lengthscale_objective = 0.3
    kernel_lengthscale_safety = 0.3
    output_scale_bjective = 50
    output_scale_safety = 50
    observation_noise_varaince = 0.05
    beta_squared = 4

    gp_model_objective = helpers.get_gp_model(
        gp_prior_mean, kernel_lengthscale_objective, output_scale_bjective, observation_noise_varaince, beta_squared)

    gp_model_safety = helpers.get_gp_model(
        gp_prior_mean, kernel_lengthscale_safety, output_scale_safety, observation_noise_varaince, beta_squared)

    domain = [(4., 8.), (-1.7, -0.6)]
    latent_function = evaluate_objective_and_constraint

    safe_seed = torch.zeros(1, len(domain))
    safe_seed[0, 0] = 6.9
    safe_seed[0, 1] = -0.9
    safe_seed_value_objective, safe_seed_value_safety = latent_function(safe_seed)
    gp_model_objective.add_observations(safe_seed, safe_seed_value_objective)
    gp_model_safety.add_observations(safe_seed, safe_seed_value_safety)
    gp_model_safety.add_observations(safe_seed, safe_seed_value_safety)
    gp_model_safety.add_observations(safe_seed, safe_seed_value_safety)

    method = parameters["method"]
    number_of_samples_per_dimension = 200
    number_of_samples = np.power(number_of_samples_per_dimension, len(domain))
    if method == 'safe_opt':
        lipschitz_constant = parameters["lipschitz_constant"]
        acquisition = SafeOptAcquisition(gp_model_safety, gp_model_objective, safe_seed, domain, lipschitz_constant,
                                         number_of_samples)
    elif method == 'ise_mes':
        learning_rate = parameters["learning_rate"]
        epochs = parameters["epochs"]
        acquisition = InfoTheoreticSafeBoAcquisition(
            gp_model_safety, gp_model_objective, domain, safe_seed, number_of_samples, learning_rate, epochs, 50)
    elif method == 'mes_safe':
        candidate_set = torch.rand(50, len(domain), dtype=torch.tensor(domain).dtype)
        candidate_set = torch.tensor(domain)[0, 0] + \
                        (torch.tensor(domain)[0, 1] - torch.tensor(domain)[0, 0]) * candidate_set
        pure_mes_acquisition = qMaxValueEntropy(gp_model_objective, candidate_set)
        acquisition = DiscreteSafeAcquisitionOptimizer(
            gp_model_safety, safe_seed, pure_mes_acquisition, domain, number_of_samples)
    elif method == 'mes':
        acquisition = MesAcquisitionWrapper(gp_model_objective, domain)
    else:
        raise ValueError(f'Requested method "{method}" not available!')

    number_of_iterations = 200
    max_value = -100
    for i in range(number_of_iterations):
        next_point_to_sample, acquisition_value = acquisition.optimize()

        noisy_observation_objective, noisy_observation_safety = latent_function(
            next_point_to_sample, observation_noise_varaince)
        gp_model_objective.add_observations(next_point_to_sample, noisy_observation_objective)
        gp_model_safety.add_observations(next_point_to_sample, noisy_observation_safety)

        next_eval_poin_objective_value = latent_function(next_point_to_sample)[0].item()
        max_value = next_eval_poin_objective_value if next_eval_poin_objective_value > max_value else max_value

        # log metrics


if __name__ == "__main__":
    run_pd_controller_experiment(default_pd_controller_exp_parameters)
