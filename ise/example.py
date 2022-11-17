# Copyright (c) 2022 Robert Bosch GmbH
# Author: Alessandro G. Bottero
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

from gpytorch import kernels
import matplotlib.pyplot as plt
import numpy as np
import torch

from ise.acquisitions.ise_acquisition import IseAcquisition
from ise.gaussian_processes.gaussian_noise_gp import GaussianNoiseGP
from ise.utils.one_dimensional_plotting_utils import OneDGPPlotInfos, get_1d_safe_set, plot_data_and_posterior
from ise.utils.generic_utils import get_gp_sample_safe_at_origin

def main():
    domain = [(-3., 4.)]
    kernel_lengthscale = 0.9
    gp_prior_mean = 0
    output_scale = 1
    observation_noise_variance = 0.005
    beta_squared = 4

    # Define GP model hyperparameters and instantiate GP model
    gp_hyperparameters = {
        'covar_module.base_kernel.lengthscale': torch.tensor(kernel_lengthscale),
        'covar_module.outputscale': torch.tensor(output_scale),
        'mean_module.constant': torch.tensor(gp_prior_mean),
        'likelihood.noise_covar.noise': torch.tensor(observation_noise_variance),
    }
    gp_model = GaussianNoiseGP(
        gp_hyperparameters, beta_squared, kernels.RBFKernel(), torch.tensor([]), torch.tensor([]))

    # Sample the unknown constraint function from the GP model
    latent_function = get_gp_sample_safe_at_origin(gp_model, domain, observation_noise_variance)

    # Define safe seed and add its value (without noise) to GP database
    safe_seed = torch.zeros(1, len(domain))
    safe_seed_value = latent_function(safe_seed, False)
    gp_model.add_observations(safe_seed, safe_seed_value)

    # Instantiate ISE acquisition function
    learning_rate = 0.01
    learning_epochs = 100
    acquisition_function = IseAcquisition(gp_model, safe_seed, domain, learning_rate, learning_epochs)

    # Create object collecting information needed for plotting
    latent_function_safe_set = get_1d_safe_set(latent_function, kernel_lengthscale, domain[0])
    plot_info = OneDGPPlotInfos(
        gp_model, latent_function, domain[0], latent_function_safe_set, [], None, np.sqrt(beta_squared))

    # For given number of iterations optimize acquisition function, evaluate at found optimum and plot resulting
    # GP posterior
    number_of_iterations = 20
    for _ in range(number_of_iterations):
        next_point_to_sample, acquisition_value = acquisition_function.optimize()

        plot_info.next_evaluation_point = next_point_to_sample
        plot_info.current_safe_set = get_1d_safe_set(gp_model.lower_confidence_bound, kernel_lengthscale, domain[0])
        plot_data_and_posterior(plot_info)
        plt.show()

        noisy_observation = latent_function(next_point_to_sample)
        gp_model.add_observations(next_point_to_sample, noisy_observation)


if __name__ == '__main__':
    random_seed = 42
    torch.manual_seed(random_seed)
    main()
