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

import math
import torch

from ise.acquisitions.line_bo_acquisiton_base import LineBoAcquisitionBase
from ise.acquisitions.heuristic_stage_opt_like_acquisition import HeuristicStageOptAcquisition
from ise.utils.generic_utils import sample_uniform_in_box, point_is_within_box

# TODO: Address code duplications with StageOptLineBoAcquisition
class HeuristicStageOptLineBoAcquisition(LineBoAcquisitionBase):
    def __init__(self, gp_model, safe_seed, domain, lipschitz_constant, objective, heterosketastic=False):
        '''
        Constructor
        '''
        super().__init__(gp_model, safe_seed, domain, objective)
        self.__lipschitz_constant = lipschitz_constant
        self.__heuristic_acquisition = HeuristicStageOptAcquisition(
            gp_model, safe_seed, domain, lipschitz_constant, 1, heterosketastic)


    def _compute_safe_and_unsafe_sets(self, points):
        '''
        Separate the given set of points into safe and unsafe sets
        Parameters
        ----------
        points (torch.Tensor): points to be separated into safe and unsafe ones

        Returns
        -------
        (pair of torch.Tensor) the safe and unsafe subsets of points
        '''

        points_are_safe = (self._model.lower_confidence_bound(points) >= 0).squeeze()
        points_are_unsafe = torch.logical_not(points_are_safe)

        return points[points_are_safe], points[points_are_unsafe]


    def _find_argmax_location(self, origin, normalized_direction):
        '''
        Finds points that maximizes acquisition function on line passing through origin and
        with direction normalized_direction

        Parameters
        ----------
        origin (torch.Tensor): Origin of the line on which the acquisition function has to be optimized
        normalized_direction: (torch.Tensor)  Direction of the line on which the acquisition function has
        to be optimized

        Returns
        -------
        (torch.Tensor) Point that maximises acquisition function along given line
        '''

        standard_subspace_bounds = self._get_subspace_bounds(origin, normalized_direction)
        subspace_length = standard_subspace_bounds[0][1] - standard_subspace_bounds[0][0]
        samples_per_unit_length = 50
        number_of_samples = math.ceil(subspace_length * samples_per_unit_length)

        subspace_augmented_samples = sample_uniform_in_box(standard_subspace_bounds, number_of_samples)

        rembedded_samples = self._one_d_samples_to_full_domain(subspace_augmented_samples, origin, normalized_direction)
        samples_are_in_domain = point_is_within_box(rembedded_samples, self._domain)
        rembedded_samples = rembedded_samples[samples_are_in_domain].unsqueeze(1)
        if len(rembedded_samples) == 0:
            return None, None

        safe_samples, unsafe_samples = self._compute_safe_and_unsafe_sets(rembedded_samples)
        if len(safe_samples) == 0:
            return None, None

        return self.__heuristic_acquisition.optimize((safe_samples, unsafe_samples))


    def optimize(self):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain

        Returns
        -------
        (torch.Tensor) Point that maximizes acquisition function on a sample of random 1d subsets
        '''

        return self._optimize(self._find_argmax_location)
