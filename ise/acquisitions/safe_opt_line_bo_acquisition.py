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

import math
import torch

from ise.acquisitions.line_bo_acquisiton_base import LineBoAcquisitionBase
from ise.acquisitions.safe_opt_acquisition import SafeOptAcquisition
from ise.utils.generic_utils import sample_uniform_in_box, point_is_within_box

class SafeOptLineBoAcquisition(LineBoAcquisitionBase):
    def __init__(self, gp_model_safety, gp_model_objective, safe_seed, domain, lipschitz_constant):
        '''
        Constructor

        Parameters
        ----------
        gp_model_safety (gpytorch.models.ExactGP): GP that models safety constraint function
        gp_model_objective (gpytorch.models.ExactGP): GP that models objective function
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        lipschitz_constant (float): Lipschitz constant to be used by the acquisition function
        objective (callable): ojective function modeled by the GP
        '''
        
        super().__init__(safe_seed, domain)
        self._model_safety = gp_model_safety
        self._model_objective = gp_model_objective
        self._lipschitz_constant = lipschitz_constant
        self._safeopt_acquisition = SafeOptAcquisition(
            gp_model_safety, gp_model_objective, safe_seed, domain, lipschitz_constant, 1)


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
        
        points_are_safe = (self._model_safety.lower_confidence_bound(points) >= 0).squeeze()
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
        samples_per_unit_length = 100
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

        return self._safeopt_acquisition.optimize((safe_samples, unsafe_samples))


    def optimize(self):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain

        Returns
        -------
        (torch.Tensor) Point that maximizes acquisition function on a sample of random 1d subsets
        '''

        return self._optimize(self._find_argmax_location)
