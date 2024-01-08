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

import copy
import math
import numpy as np
import torch

from ise.utils.optimization_utils import IseConstrainedOptimizer
from ise.utils.generic_utils import sample_uniform_in_box


class IseAcquisition:
    def __init__(
            self,
            gp_model,
            safe_seed,
            domain,
            learning_rate,
            learning_epochs,
            number_of_samples=None,
            noise_variance=None):
        '''
        Constructor

        Parameters
        ----------
        gp_model (GaussianNoiseGP): GP model the acquisition function is associated to
        safe_seed (torch.Tensor): initial known safe point
        domain (list of float pairs [(min_0, max_0), ..., (min_d, max_d)]): input domain in the form
        of a d-dimensional box
        learning_rate (float): initial learning rate for the gradient ascent optimization
        learning_epochs (int): gradient ascent iterations number
        number_of_samples (int): number of samples to use to find optimization starting points
        noise_variance (callable): optional, function that maps a point in the domain to its observation noise.
        To be provided in case of heteroskedastik GP
        '''

        self._model = gp_model
        self._domain = domain
        self._safe_seed = safe_seed
        self._dimension = len(domain)
        self._c_1 = 1. / (np.pi * np.log(2))
        self._learning_rate = learning_rate
        self._learning_epochs = learning_epochs
        self._relevant_lengthscale = gp_model.covar_module.base_kernel.lengthscale.item()
        self._search_radius = 1.1 * self._relevant_lengthscale
        self._number_of_samples = number_of_samples
        if noise_variance is not None:
            self._noise_variance = noise_variance
        else:
            self._noise_variance = self._model.likelihood.noise_covar.noise


    def _update_search_radius(self, next_evaluation_point):
        '''
        Updates the search radius increasing it if needed.

        Parameters
        ----------
        next_evaluation_point (torch.Tensor): point that will be evaluated next
        '''

        safe_seed_next_point_distance = (self._safe_seed - next_evaluation_point).norm().item()
        new_search_distance_proposal = safe_seed_next_point_distance + self._relevant_lengthscale
        if new_search_distance_proposal > self._search_radius:
            self._search_radius = new_search_distance_proposal


    def _psi_entropy(self, mean_variance_ratio):
        '''
        Computes entropy of \Psi at a point with \mu^2/\sigma^2 = mean_variance_ratio

        Parameters
        ----------
        mean_variance_ratio (torch.Tensor): ratio of posterior mean squared and variance

        Returns
        -------
        (torch.Tensor) entropy of Psi
        '''

        return np.log(2) * torch.exp(-self._c_1 * mean_variance_ratio)


    def _average_post_measurement_psi_entropy(
            self, entropy_evaluation_point_mean_variance_ratio, observed_point_variance, correlation, noise_variance):
        '''
        Computes average Psi entropy at a parameter after an evaluation at another parameter
        
        Parameters
        ----------
        entropy_evaluation_point_mean (torch.Tensor): posterior mean at the point of the hypothetical measurement
        observed_point_variance (torch.Tensor): posterior variance at the point of the hypothetical measurement
        correlation (torch.Tensor): correlation between the point of the hypothetical measurement and
        the point where we compute the average Psi entropy after the measurement
        
        Returns
        -------
        (torch.Tensor) average Psi entropy after an hypothetical measurement
        '''

        rho_nu = observed_point_variance / (noise_variance + observed_point_variance)

        rhos_product = rho_nu * torch.pow(correlation, 2)
        rhos_product[rhos_product > 1.] = 1.  # Protects from negative square root argument in case of numerical errors

        c_2 = 2 * self._c_1 - 1
        common_rho_factor = 1 + c_2 * rhos_product
        square_root_argument = (1 - rhos_product) / common_rho_factor
        exponent_argument = -self._c_1 * entropy_evaluation_point_mean_variance_ratio / common_rho_factor

        return np.log(2) * torch.sqrt(square_root_argument) * torch.exp(exponent_argument)


    def compute_acquisition_value(self, observed_point, point_to_evaluate_entropy_variance_at):
        '''
        Compute the value of the acquisition function, i.e. the expected entropy variation at
        point_to_evaluate_entropy_variance_at after an hypothetical measurement at observed_point
        
        Parameters
        ----------
        observed_point (torch.tensor): Point at which the hypothetical measurement of the
        latent function is collected
        point_to_evaluate_entropy_variance_at (torch.tensor): point at which to compute the
        expected Psi entropy variation
        
        Returns
        -------
        (torch.Tensor) Average Psi entropy variation at point_to_evaluate_entropy_variance_at
        after a measurement at observed_point
        '''

        catted_inputs = torch.cat((observed_point, point_to_evaluate_entropy_variance_at), 1)
        posterior_gp = self._model(catted_inputs)
        covariance_matrix = posterior_gp.lazy_covariance_matrix
        with torch.no_grad():
            if callable(self._noise_variance):
                noise = self._noise_variance(observed_point)
            else:
                noise = torch.ones((len(observed_point), 1)) * self._noise_variance

        noise_at_evaluation_point = noise[..., 0]

        entropy_evaluation_point_mean = posterior_gp.mean[..., 1]
        entropy_evaluation_point_variance = covariance_matrix[..., 1][..., 1].unsqueeze(0)
        entropy_evaluation_point_mean_variance_ratio = \
            torch.pow(entropy_evaluation_point_mean, 2) / entropy_evaluation_point_variance

        observed_point_variance = covariance_matrix[..., 0][..., 0].unsqueeze(0)
        covariance = covariance_matrix[..., 0][..., 1]
        correlation = covariance / (torch.sqrt(observed_point_variance) * torch.sqrt(entropy_evaluation_point_variance))

        pre_observarion_entropy = self._psi_entropy(entropy_evaluation_point_mean_variance_ratio)
        post_observarion_average_entropy = self._average_post_measurement_psi_entropy(
            entropy_evaluation_point_mean_variance_ratio,
            observed_point_variance,
            correlation,
            noise_at_evaluation_point)

        lower_confidence_bound_at_observed_point = \
            posterior_gp.mean[..., 0] - torch.sqrt(self._model.beta_squared * observed_point_variance)
        sign_of_safety_condition = torch.sign(lower_confidence_bound_at_observed_point).squeeze()
        sign_of_safety_condition[sign_of_safety_condition == 0] = 1.

        return sign_of_safety_condition * (pre_observarion_entropy - post_observarion_average_entropy)

    def _points_are_safe(self, points):
        '''
        Checks whether or not points are all in the safe set
        
        Parameters
        ----------
        points (torch.Tensor): ponts to check

        Returns
        -------
        (bool) True if all points are in the safe set, False Otherwise
        '''

        points_are_unsafe = self._model.lower_confidence_bound(points) < 0
        return not points_are_unsafe.any()

    def _select_optimization_starting_points(self, number_of_staring_points):
        '''
        Identifies promising starters for the optimization procedure
        
        Parameters
        ----------
        number_of_staring_points: (int) number of points that the function will return
        
        Returns
        -------
        :(torch.Tensor) number_of_staring_points points that realize the greatest acquisition
        among the sampled ones
        '''

        potential_starters_x = sample_uniform_in_box(self._domain, self._number_of_samples).unsqueeze(1)
        potential_starters_x = torch.cat((potential_starters_x, self._safe_seed.unsqueeze(0)))
        potential_starters_z = copy.deepcopy(potential_starters_x)
        
        number_of_samples_to_compute_in_parallel = 1000
        number_of_batches = math.ceil(self._number_of_samples / number_of_samples_to_compute_in_parallel)
        potential_starters_values = torch.empty(0)
        with torch.no_grad():
            for i in range(number_of_batches):
                start_index = i * number_of_samples_to_compute_in_parallel
                end_index = start_index + number_of_samples_to_compute_in_parallel if i < number_of_batches - 1 else None
                values_of_current_batch = self.compute_acquisition_value(
                    potential_starters_x[start_index: end_index],
                    potential_starters_z[start_index: end_index]).squeeze()
                if values_of_current_batch.dim() == 0:
                    values_of_current_batch.unsqueeze(0)
                potential_starters_values = torch.cat((potential_starters_values, values_of_current_batch))

        max_values_indices = torch.topk(potential_starters_values, number_of_staring_points)[1]
        starters_x = potential_starters_x[max_values_indices]
        
        if not self._points_are_safe(starters_x):
            return self._select_optimization_starting_points(number_of_staring_points)
        
        return starters_x.detach(), potential_starters_z[max_values_indices].detach()


    def optimize(self, starting_points=None):
        '''
        Optimizes the acquisition function using gradient ascent and multiple restarts
        
        Parameters
        ----------
        starting_points: (pair of tensors) (points_x, points_z) to use as starters for optimization.
        If not provided random points will be selected

        Returns
        -------
        (torch.Tensor) The point resulting from the gradient ascent procedure
        '''

        number_of_starting_points = 5
        if starting_points is None:
            starting_points_x, starting_points_z = self._select_optimization_starting_points(number_of_starting_points)
        else:
            starting_points_x, starting_points_z = starting_points
            potential_starters_values = self.compute_acquisition_value(starting_points_x, starting_points_z)
            number_of_starters = potential_starters_values.size()[1]
            number_of_starting_points = number_of_starting_points if \
                number_of_starters > number_of_starting_points else number_of_starters
            max_values_indices = torch.topk(potential_starters_values, number_of_starting_points)[1]
            starting_points_x = starting_points_x[max_values_indices].squeeze(0)
            starting_points_z = starting_points_z[max_values_indices].squeeze(0)

        starting_points_x.requires_grad = True
        starting_points_z.requires_grad = True

        optimizer = IseConstrainedOptimizer(
            starting_points_x,
            starting_points_z,
            self._domain,
            self._learning_rate,
            self._model.lower_confidence_bound)

        for _ in range(self._learning_epochs):
            acquisition_value = self.compute_acquisition_value(starting_points_x, starting_points_z).sum()
            acquisition_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        values_of_found_optima = self.compute_acquisition_value(starting_points_x, starting_points_z)
        index_of_point_with_biggest_value = torch.topk(values_of_found_optima, 1)[1]
        if values_of_found_optima.shape[1] > 1:
            values_of_found_optima = values_of_found_optima.squeeze()

        found_optimum = starting_points_x[index_of_point_with_biggest_value].view(1, self._dimension).detach()
        self._update_search_radius(found_optimum)

        return found_optimum, values_of_found_optima[index_of_point_with_biggest_value].detach()
