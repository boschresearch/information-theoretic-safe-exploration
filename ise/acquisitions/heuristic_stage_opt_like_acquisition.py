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

import gpytorch
import math
import numpy as np
import torch

from ise.utils.generic_utils import sample_uniform_in_box, point_is_within_box, get_linearly_spaced_combinations

# TODO: Address code duplication with StageOptAcquisition
class HeuristicStageOptAcquisition():
    def __init__(self, gp_model, safe_seed, domain, size_of_discrete_domain, heteroskedastic=False, grid_domain=False):
        '''
        Constructor

        Parameters
        ----------
        gp_model (gpytorch.models.ExactGP): GP model whose posterior is used by the acquisition function
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        size_of_discrete_domain (int): number of points to discretize the domain in
        heteroskedastic (bool) whether or not the observation process is heteroskedastic or not
        grid_domain (bool): whether or not the discretized domain has to be a equspaced grid of domain
        '''

        self._model = gp_model
        self._safe_seed = safe_seed
        self._domain = domain
        self._discrete_domain = self._discretize_domain(domain, size_of_discrete_domain, grid_domain)
        self._heteroskedastic = heteroskedastic


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


    def _point_optimistically_elarges_safe_set(self, safe_set_point, unsafe_points):
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

        upper_confidence_bound = self._model.upper_confidence_bound(safe_set_point)
        if self._heteroskedastic:
            model_with_optimistic_observation = self._model.get_fantasy_model(
                safe_set_point, upper_confidence_bound, noise=torch.tensor([0.0001]))
        else:
            model_with_optimistic_observation = self._model.get_fantasy_model(safe_set_point, upper_confidence_bound)

        return model_with_optimistic_observation.lower_confidence_bound(unsafe_points) >= 0


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

        number_of_samples_to_compute_in_parallel = 1000
        number_of_total_points = len(unsafe_points)
        number_of_batches = math.ceil(number_of_total_points / number_of_samples_to_compute_in_parallel)
        safe_points_enlarge_safe_set = torch.empty(0)
        for i in range(number_of_batches):
            start_index = i * number_of_samples_to_compute_in_parallel
            end_index = start_index + number_of_samples_to_compute_in_parallel if i < number_of_batches - 1 else None
            current_batch_values = self._point_optimistically_elarges_safe_set(
                safe_point, unsafe_points[start_index: end_index]).squeeze()
            if current_batch_values.dim() == 0:
                current_batch_values = current_batch_values.unsqueeze(0)
            safe_points_enlarge_safe_set = torch.cat((safe_points_enlarge_safe_set, current_batch_values))

        return torch.any(safe_points_enlarge_safe_set)


    def _compute_acquisition_value(self, point):
        '''
        Compute the value of the acquisition function at the specified parameter(s)

        Parameters
        ----------
        point (torch.Tensor): point(s) at which to compute the acquisition value

        Returns
        -------
        (torch.Tensor) Acquisition value at point
        '''

        return self._model.posterior(point).variance


    def _compute_safe_and_unsafe_sets(self):
        '''
        Separate the domain into safe and unsafe sets

        Returns
        -------
        (pair of torch.Tensor) the safe and unsafe parts of the discretized domain
        '''

        points_are_safe = (self._model.lower_confidence_bound(self._discrete_domain) >= 0).squeeze()
        points_are_unsafe = torch.logical_not(points_are_safe)
        safe_set = self._discrete_domain[points_are_safe]
        safe_set = torch.cat((safe_set, self._safe_seed.view(1, 1, len(self._domain))))

        return safe_set, self._discrete_domain[points_are_unsafe]


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

        acquisition_values = self._compute_acquisition_value(points)
        return torch.sort(acquisition_values.squeeze(), descending=True)[1]


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
            indices_for_sorted_safe_points = self._get_indices_sorted_according_to_acquisition(safe_set)
            dimensions = len(self._domain)
            if len(safe_set) == 1:
                indices_for_sorted_safe_points = torch.tensor([0])
            for index in indices_for_sorted_safe_points:
                safe_point = safe_set[index].view(1, 1, dimensions)
                if self._safe_point_is_expander(safe_point, unsafe_set):
                    return safe_point.squeeze(0), self._compute_acquisition_value(safe_point.squeeze(0))

            random_safe_point = safe_set[torch.randint(0, len(safe_set), (1,))].squeeze(0)
            return random_safe_point, self._compute_acquisition_value(random_safe_point)
