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

from gpytorch import kernels
import torch

from ise.gaussian_processes.gaussian_noise_gp import GaussianNoiseGP

def get_gp_model(prior_mean, lengthscale, output_scale, noise_variance, beta):
    gp_hyperparameters = {
        'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
        'covar_module.outputscale': torch.tensor(output_scale),
        'mean_module.constant': torch.tensor(prior_mean)
    }

    if not callable(noise_variance):
        gp_hyperparameters['likelihood.noise_covar.noise'] = torch.tensor(noise_variance),
        return GaussianNoiseGP(gp_hyperparameters, beta, kernels.MaternKernel(), torch.tensor([]), torch.tensor([]))

    return GaussianNoiseGP(gp_hyperparameters, beta, kernels.RBFKernel(), torch.tensor([]), torch.tensor([]), True)
