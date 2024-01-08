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

import torch

from ise.utils import generic_utils

class LineBoAcquisitionBase:
    def __init__(self, safe_seed, domain):
        '''
        Constructor

        Parameters
        ----------
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        '''

        self._safe_seed = safe_seed
        self._last_sampled_point = safe_seed
        self._domain = domain

    def _get_subspace_bounds(self, origin, normalized_direction):
        '''
        Compute how much one can move along normalized_direction starting from origin while still remaining
        inside the domain
        
        Parameters
        ----------
        origin (torch.Tensor): origin point from which to compute the maximum allowed displacement
        normalized_direction (torch.Tensor): vector defining units of dispacements from origin

        Returns
        -------
        (pair of torch.Tensor [(a, b)]) respectively maximum negative and positive multiples of normalized_direction
        that can be added to origin and still remaining inside the domain
        '''
        
        maximum_displacements_from_origin = []
        origin = origin.squeeze()
        normalized_direction = normalized_direction.squeeze()
        for dimension in range(len(self._domain)):
            if normalized_direction[dimension] == 0:
                continue
            maximum_left_displacement = \
                (self._domain[dimension][0] - origin[dimension]) / normalized_direction[dimension]
            maximum_right_displacement = \
                (self._domain[dimension][1] - origin[dimension]) / normalized_direction[dimension]
            maximum_displacements_from_origin.append(maximum_left_displacement)
            maximum_displacements_from_origin.append(maximum_right_displacement)
        maximum_displacements_from_origin = torch.tensor(maximum_displacements_from_origin)

        positive_displacements = maximum_displacements_from_origin[maximum_displacements_from_origin > 0]
        negative_displacements = maximum_displacements_from_origin[maximum_displacements_from_origin < 0]

        return [(torch.max(negative_displacements).item(), torch.min(positive_displacements).item())]


    def _one_d_samples_to_full_domain(self, one_d_samples, origin, direction):
        '''
        Re-embed one-dimensional point(s) along line passing through origin and with direction direction within the
        full d-dimensional domain
        
        Parameters
        ----------
        one_d_samples (torch.Tensor): 1-d coordinates of samples to re-embed in full domain. The coordinates are the
        number 'k' in 'sample = origin + k * direction'
        origin (torch.Tensor): Origin of the 1-d subspace the samples belong to
        direction (torch.Tensor): Direction of the 1-d subspace the samples belong to

        Returns
        -------
        (torch.Tensor): Re-embedded one_d_samples as d-dimensional points
        '''
        
        return origin + direction * one_d_samples


    def _get_normalized_direction(self, origin, point_for_direction):
        '''
        Computes the normalized direction of the line passing through origin and point_for_direction
        
        Parameters
        ----------
        origin (torch.Tensor): One of the two points used to calculate the direction
        point_for_direction (torch.Tensor): The other point used to calculate the direction

        Returns
        -------
        (torch.Tensor): Normalized vector representing the direction of the line passing through origin and
        point_for_direction
        '''

        direction = (point_for_direction - origin)
        return direction / torch.norm(direction)


    def _optimize(self, find_argmax_location):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain
        
        Parameters
        ----------
        find_argmax_location (callable) function that optimizes a  custom objective along a 1-d line

        Returns
        -------
        (torch.Tensor) A point in the domain that maximises the acquisition function within the analyzed 1-d subspaces
        '''

        argmaxs = []
        argmaxs_values = []
        while len(argmaxs) < 5:
            origins = [self._safe_seed, self._last_sampled_point]
            for _ in range(5):
                origins.append(generic_utils.sample_uniform_in_box(self._domain, 1))
            point_for_direction = generic_utils.sample_uniform_in_box(self._domain, 1)
            for origin in origins:
                normalized_direction = self._get_normalized_direction(origin, point_for_direction)
                argmax, value = find_argmax_location(origin, normalized_direction)
                if argmax is not None:
                    argmaxs.append(argmax)
                    argmaxs_values.append(value)

        argmaxs_values = torch.tensor(argmaxs_values)
        optimum_value_index = torch.topk(argmaxs_values, 1)[1]

        self._last_sampled_point = argmaxs[optimum_value_index]

        return argmaxs[optimum_value_index], argmaxs_values[optimum_value_index]
