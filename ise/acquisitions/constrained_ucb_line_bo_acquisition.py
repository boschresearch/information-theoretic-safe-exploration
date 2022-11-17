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
import numpy as np
import torch

from ise.acquisitions.line_bo_acquisiton_base import LineBoAcquisitionBase
from ise.utils import generic_utils

class ConstrainedUcbLineBoAcquisition(LineBoAcquisitionBase):
    def __init__(self, gp_model, safe_seed, domain, constraint, objective):
        '''
        Constructor

        Parameters
        ----------
        gp_model (gpytorch.models.ExactGP): GP model whose posterior is used by the acquisition function
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        constraint (callable): Constraint f (of the form f(x) >= 0) that the point to sample must satisfy
        objective (callable): ojective function modeled by the GP
        '''

        super().__init__(gp_model, safe_seed, domain, objective)
        self._constraint = constraint
        self._beta = np.sqrt(gp_model.beta_squared)


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

        posterior_gp = self._model(point)
        mean = posterior_gp.mean
        standard_deviation = torch.sqrt(posterior_gp.variance)
        acquisition_value = (mean + self._beta * standard_deviation).view((1, len(point)))

        sign_of_safety_condition = \
            torch.sign(self._constraint(point)).squeeze() * torch.sign(acquisition_value).squeeze()

        return sign_of_safety_condition * acquisition_value


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

        standard_subspace_bounds = self._get_subspace_bounds(self._last_sampled_point, normalized_direction)
        subspace_length = standard_subspace_bounds[0][1] - standard_subspace_bounds[0][0]
        samples_per_unit_length = 35
        number_of_samples = math.ceil(subspace_length * samples_per_unit_length)

        subspace_augmented_samples = generic_utils.sample_uniform_in_box(standard_subspace_bounds, number_of_samples)

        rembedded_samples = self._one_d_samples_to_full_domain(subspace_augmented_samples, origin, normalized_direction)
        samples_are_in_domain = generic_utils.point_is_within_box(rembedded_samples, self._domain)
        rembedded_samples = rembedded_samples[samples_are_in_domain].unsqueeze(1)
        if len(rembedded_samples) == 0:
            return None, None

        samples_values = self._compute_acquisition_value(rembedded_samples).squeeze()

        samples_are_safe = samples_values > 0.  # Assuming objective is negative when constraint is violated
        rembedded_samples = rembedded_samples[samples_are_safe]
        if len(rembedded_samples) == 0:
            return None, None

        safe_values = samples_values[samples_are_safe]
        optimum_value_index = torch.topk(safe_values, 1)[1]

        return rembedded_samples[optimum_value_index].view(1, len(self._domain)), safe_values[optimum_value_index]


    def optimize(self):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain

        Returns
        -------
        (torch.Tensor) Point that maximizes acquisition function on a sample of random 1d subsets
        '''

        return self._optimize(self._find_argmax_location)
