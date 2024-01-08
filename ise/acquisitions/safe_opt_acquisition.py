# Copyright (c) 2023 Robert Bosch GmbH
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

import gpytorch
import math
import numpy as np
import torch

from ise.acquisitions.discrete_safe_acquisition_optimizer import DiscreteSafeAcquisitionOptimizer
from ise.utils.generic_utils import sample_uniform_in_box, point_is_within_box, get_linearly_spaced_combinations

class SafeOptAcquisition():
    def __init__(self,
                 gp_model_safety,
                 gp_model_ojective,
                 safe_seed,
                 domain,
                 lipschitz_constant,
                 size_of_discrete_domain,
                 grid_domain=False):
        '''
        Constructor

        Parameters
        ----------
        gp_model_safety (gpytorch.models.ExactGP): GP that models safety constraint function
        gp_model_objective (gpytorch.models.ExactGP): GP that models objective function
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        lipschitz_constant (float): Lipschitz constant to be used by the acquisition function
        size_of_discrete_domain (int): number of points to discretize the domain in
        grid_domain (bool): whether the discretized domain has to be a equspaced grid of domain
        '''

        self._constraint_model = gp_model_safety
        self._objective_model = gp_model_ojective
        self._safe_seed = safe_seed
        self._domain = domain
        self._discrete_domain = self._discretize_domain(domain, size_of_discrete_domain, grid_domain)
        self._lipschitz_constant = lipschitz_constant


    def _discretize_domain(self, domain, size_of_discrete_domain, grid_domain):
        '''
        Create a discretization of the provided continuous domain

        Parameters
        ----------
        domain: (list of pairs of floats): list of the coordinates of the domain's box vertices
        size_of_discrete_domain (int): number of points contained in the discretization of the domain
        grid_domain (bool) whether or not the discretization should be a equispaced grid rather than
        uniform random points

        Returns
        -------
        (torch.Tensor) collection of points composing the discretized domain
        '''
        
        if grid_domain:
            dimensions = len(domain)
            number_of_samples_per_dimension = math.floor(np.power(size_of_discrete_domain, 1. / dimensions))
            grid = get_linearly_spaced_combinations(domain, number_of_samples_per_dimension)
            return torch.tensor(grid, dtype=torch.float32).view((len(grid), 1, len(domain)))

        discrete_domain = sample_uniform_in_box(domain, size_of_discrete_domain)
        points_within_domain = discrete_domain[point_is_within_box(discrete_domain, domain)]
        discrete_domain = points_within_domain.view((len(points_within_domain), 1, len(domain)))

        return discrete_domain


    def _compute_kernel_distance(self, model, x, y):
        '''
        Compute the distance between two points according to the metric of the kernel of the underlying GP
        
        Parameters
        ----------
        x (torch.Tensor): One of the two points
        y (torch.Tensor): The other point

        Returns
        -------
        (torch.Tensor) The distace between x and y

        '''
        
        with gpytorch.settings.prior_mode(True):
            if x.shape != y.shape:
                x = torch.ones(y.shape) * x
            catted_inputs = torch.cat((x, y), 1)
            covariance_matrix = model(catted_inputs).lazy_covariance_matrix
            covariance = covariance_matrix[..., 0][..., 1].unsqueeze(0)
            variance_x = covariance_matrix[..., 0][..., 0].unsqueeze(0)
            variance_y = covariance_matrix[..., 1][..., 1].unsqueeze(0)
            return torch.sqrt(variance_x + variance_y - 2 * covariance)

    def _point_optimistically_elarges_safe_set(self, safe_set_point, unsafe_point):
        '''
        Checks whether or not an evaluation at safe_set_point optimistically makes unsafe_point safe
 
        Parameters
        ----------
        safe_set_point (torch.Tensor): The safe set point which we want to test
        unsafe_point (torch.Tensor): The unsafe point that may be made safe by an evaluation at safe_set_point

        Returns
        -------
        (bool) whether or not observing f at safe_set_point optimistically makes unsafe_point safe
        '''
        
        upper_confidence_bound = self._constraint_model.upper_confidence_bound(safe_set_point).squeeze()
        distance = self._compute_kernel_distance(self._constraint_model, safe_set_point, unsafe_point).squeeze()

        return upper_confidence_bound - self._lipschitz_constant * distance >= 0


    def _safe_point_is_expander(self, safe_point, unsafe_points):
        '''
        Checks whether or not evaluating safe_point optimistically makes any of the unsafe_points safe

        Parameters
        ----------
        safe_point (torch.Tensor): safe point to be checked
        unsafe_points (torch.Tensor): points that might be made safe by evaluating safe_point

        Returns
        -------
        Whether or not any of unsafe_points is optimistically made safe by an evaluation at safe_point
        '''

        safe_point = torch.ones(unsafe_points.shape) * safe_point

        number_of_samples_to_compute_in_parallel = 1000
        number_of_total_points = len(safe_point)
        number_of_batches = math.ceil(number_of_total_points / number_of_samples_to_compute_in_parallel)
        safe_points_enlarge_safe_set = torch.empty(0)
        with torch.no_grad():
            for i in range(number_of_batches):
                start_index = i * number_of_samples_to_compute_in_parallel
                end_index = \
                    start_index + number_of_samples_to_compute_in_parallel if i < number_of_batches - 1 else None
                current_batch_values = self._point_optimistically_elarges_safe_set(
                    safe_point[start_index: end_index], unsafe_points[start_index: end_index]).squeeze()
                if current_batch_values.dim() == 0:
                    current_batch_values = current_batch_values.unsqueeze(0)
                safe_points_enlarge_safe_set = torch.cat((safe_points_enlarge_safe_set, current_batch_values))

            return torch.any(safe_points_enlarge_safe_set)


    def _compute_max_safe_set_lower_bound(self, safe_set):
        def proxy_acquisition_function(x):
            return self._objective_model.lower_confidence_bound(x).squeeze()

        discrete_optimizer = DiscreteSafeAcquisitionOptimizer(
            self._objective_model, self._safe_seed, proxy_acquisition_function, None, None, self._constraint_model)
        return discrete_optimizer.optimize(safe_set)[1]

    def _safe_point_is_maximizer(self, safe_point, max_safe_set_lower_bound):
        safe_point_upper_bound = self._objective_model.upper_confidence_bound(safe_point)

        return safe_point_upper_bound >= max_safe_set_lower_bound



    def _compute_acquisition_value(self, point, model):
        '''
        Compute the value of the acquisition function at the specified parameter(s)
        
        Parameters
        ----------
        point (torch.Tensor): point(s) at which to compute the acquisition value

        Returns
        -------
        (torch.Tensor) Acquisition value at point
        '''

        return model.posterior(point).variance


    def _compute_safe_and_unsafe_sets(self):
        '''
        Separate the domain into safe and unsafe sets

        Returns
        -------
        (pair of torch.Tensor) the safe and unsafe parts of the discretized domain
        '''

        points_are_safe = (self._constraint_model.lower_confidence_bound(self._discrete_domain) >= 0).squeeze()
        points_are_unsafe = torch.logical_not(points_are_safe)
        safe_set = self._discrete_domain[points_are_safe]
        safe_set = torch.cat((safe_set, self._safe_seed.view(1, 1, len(self._domain))))

        return safe_set, self._discrete_domain[points_are_unsafe]


    def _get_augmented_indices_list(self, length, extra_dim_value):
        augmented_indices = torch.ones((length, 2), dtype=torch.int) * extra_dim_value
        augmented_indices[:, 0] = torch.arange(0, length, dtype=torch.int)

        return augmented_indices


    def _get_indices_sorted_according_to_acquisition(self, points):
        '''
        Sort point according to acquisition function values

        Parameters
        ----------
        points (torch.Tensor): points to be sorted

        Returns
        -------
        (torch.Tensor) vector of indices of the sorted points

        '''

        acquisition_values_constraint = self._compute_acquisition_value(points, self._constraint_model)
        acquisition_values_objective = self._compute_acquisition_value(points, self._objective_model)
        combined_values = torch.cat((acquisition_values_constraint, acquisition_values_objective)).squeeze()
        sorted_order = torch.sort(combined_values, descending=True)[1]

        indices_constraint = self._get_augmented_indices_list(len(acquisition_values_constraint), 0)
        indices_objective = self._get_augmented_indices_list(len(acquisition_values_objective), 1)
        combined_indices = torch.cat((indices_constraint, indices_objective))

        return combined_indices[sorted_order]


    def optimize(self, sets=None):
        '''
        Compute the safe point that maximizes the acquisition function (among the expanders)

        Parameters
        ----------
        sets (torch.Tensor): Optional sets of safe and unsafe points to be used as safe and unsafe sets.
        If not provided, the discretized domain will be used to compute the safe and unsafe sets

        Returns
        -------
        (torch.Tensor) Expander point that optimizes acquisition function. If no expander is found, a safe point will
        be returned at random
        '''

        with torch.no_grad():
            if sets is None:
                safe_set, unsafe_set = self._compute_safe_and_unsafe_sets()
            else:
                safe_set, unsafe_set = sets
            max_safe_set_lower_bound = self._compute_max_safe_set_lower_bound(safe_set)
            indices_for_sorted_safe_points = self._get_indices_sorted_according_to_acquisition(safe_set)
            dimensions = len(self._domain)
            if len(safe_set) == 1:
                indices_for_sorted_safe_points = torch.tensor([[0, 0]])
            for index in indices_for_sorted_safe_points:
                safe_point = safe_set[index[0]].view(1, 1, dimensions)
                if index[1] == 0:
                    if self._safe_point_is_expander(safe_point, unsafe_set):
                        return safe_point.squeeze(0), \
                            self._compute_acquisition_value(safe_point.squeeze(0), self._constraint_model)
                if self._safe_point_is_maximizer(safe_point, max_safe_set_lower_bound):
                    return safe_point.squeeze(0), \
                        self._compute_acquisition_value(safe_point.squeeze(0), self._objective_model)

            random_safe_point = safe_set[torch.randint(0, len(safe_set), (1,))].squeeze(0)
            return random_safe_point, self._compute_acquisition_value(random_safe_point, self._constraint_model)
