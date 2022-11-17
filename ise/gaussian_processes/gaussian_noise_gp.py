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

from botorch.models.gpytorch import GPyTorchModel
import gpytorch
import torch

class GaussianNoiseGP(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1  # output dimensionality, needed by BoTorch API

    def __init__(self, hyperparameters, beta_squared, kernel, safe_seed, safe_seed_observation, heterosketastic=False):
        '''
        Constructor

        Parameters
        ----------
        hyperparameters (dictionary: {string, value}): hyperparameters for the GP model 
        beta_squared (float): Parameter which defines the size of the confidence interval, as used to define the
        safe set: S_n = {x : mean(x) - sqrt(beta) * sigma(x) > 0}
        kernel (gpytorch.kernels.Kernel object): kernel for the GP 
        safe_seed (torch.Tensor): initial point known to be safe 
        safe_seed_value (torch.Tensor): observed value at safe_seed
        heterosketastic (bool): whether or not the observation noise is heteroskedastic 
        '''
        
        self.beta_squared = beta_squared
        self.evaluated_points = safe_seed
        self.observed_values = safe_seed_observation
        self._heteroskedastic = heterosketastic
        if heterosketastic:
            self.noise_variances = torch.tensor([])

        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.tensor([])) if heterosketastic \
            else gpytorch.likelihoods.GaussianLikelihood()

        super(GaussianNoiseGP, self).__init__(
            self.evaluated_points, self.observed_values.squeeze(-1), likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

        self.initialize(**hyperparameters)
        self.eval()
        self.likelihood.eval()


    def lower_confidence_bound(self, x):
        '''
        Computes the lower confidence bound at x given the current posterior
        
        Parameters
        ----------
        x (torch.Tensor): point(s) to compute the confidence bound

        Returns
        -------
        (torch.Tensor) lower confidence bound at x
        '''

        posterior = self.posterior(x)
        return posterior.mean - torch.sqrt(self.beta_squared * posterior.variance)


    def upper_confidence_bound(self, x):
        '''
        Computes the upper confidence bound at x given the current posterior
        
        Parameters
        ----------
        x (torch.Tensor): point(s) to compute the confidence bound

        Returns
        -------
        (torch.Tensor) upper confidence bound at x
        '''

        posterior = self.posterior(x)
        return posterior.mean + torch.sqrt(self.beta_squared * posterior.variance)


    def add_observations(self, evaluation_points, observations, noises=None):
        '''
        Adds a list of observation points to the observed data
        
        Parameters
        ----------
        evaluation_points (torch.Tensor): x coordinates of observed points
        observations (torch.Tensor): observed values corresponding to the points in evaluation_points
        noises (torch.Tensor) observation noises corresponding to observations - used only if gp is heteroskedastic
        '''

        self.evaluated_points = torch.cat((self.evaluated_points, evaluation_points))
        self.observed_values = torch.cat((self.observed_values, observations))
        if self._heteroskedastic:
            self.noise_variances = torch.cat((self.noise_variances, noises))
            self.likelihood.noise = self.noise_variances

        self.set_train_data(self.evaluated_points, self.observed_values, strict=False)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
