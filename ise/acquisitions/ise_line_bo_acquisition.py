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

from ise.acquisitions.line_bo_acquisiton_base import LineBoAcquisitionBase
from ise.acquisitions.ise_acquisition import IseAcquisition
from ise.utils import generic_utils

class IseLineBoAcquisition(LineBoAcquisitionBase):
    def __init__(self, gp_model, safe_seed, domain, noise_variance=None):
        '''
        Constructor

        Parameters
        ----------
        gp_model (gpytorch.models.ExactGP): GP model whose posterior is used by the acquisition function
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        noise_variance (callable or None): Optional, to be provided in case of heteroskedastic noise it maps input
        locations to the corresponding observation noise
        '''

        super().__init__(safe_seed, domain)
        self._model = gp_model
        self._relevant_lengthscale = gp_model.covar_module.base_kernel.lengthscale.item()
        self._ise_acquisition = IseAcquisition(
            gp_model, safe_seed, domain, 0, 0, 0, noise_variance)


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

        subspace_dimensions = 1
        gaussian_mean = torch.zeros(1, 1)
        subspace_augmented_samples_x, subspace_augmented_samples_z = generic_utils.sample_from_2d_gaussian(
            subspace_dimensions,
            gaussian_mean,
            self._relevant_lengthscale,
            subspace_length,
            number_of_samples,
            standard_subspace_bounds
        )
        rembedded_samples_x = self._one_d_samples_to_full_domain(
            subspace_augmented_samples_x, origin, normalized_direction)
        rembedded_samples_z = self._one_d_samples_to_full_domain(
            subspace_augmented_samples_z, origin, normalized_direction)

        samples_are_in_domain = generic_utils.point_is_within_box(rembedded_samples_x, self._domain)
        rembedded_samples_x = rembedded_samples_x[samples_are_in_domain]
        rembedded_samples_z = rembedded_samples_z[samples_are_in_domain]
        if len(rembedded_samples_x) == 0:
            return None, None

        samples_are_safe = (self._model.lower_confidence_bound(rembedded_samples_x) >=0).squeeze()
        rembedded_samples_x = rembedded_samples_x[samples_are_safe]
        rembedded_samples_z = rembedded_samples_z[samples_are_safe]
        if len(rembedded_samples_x) == 0:
            return None, None

        number_of_samples_to_compute_in_parallel = 500
        number_of_batches = math.ceil(rembedded_samples_x.shape[1] / number_of_samples_to_compute_in_parallel)
        values = torch.empty(0)
        for i in range(number_of_batches):
            start_index = i * number_of_samples_to_compute_in_parallel
            end_index = start_index + number_of_samples_to_compute_in_parallel if i < number_of_batches - 1 else None
            values_of_current_batch = self._ise_acquisition.compute_acquisition_value(
                rembedded_samples_x[start_index: end_index].unsqueeze(1),
                rembedded_samples_z[start_index: end_index].unsqueeze(1))
            if values_of_current_batch.dim() == 0:
                values_of_current_batch.unsqueeze(0)
            values = torch.cat((values, values_of_current_batch))

        max_values_indices = torch.topk(values, 1)[1]
        return rembedded_samples_x[max_values_indices].squeeze(0), values.squeeze(0)[max_values_indices]


    def optimize(self):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain

        Returns
        -------
        (torch.Tensor) Point that maximizes acquisition function on a sample of random 1d subsets
        '''

        return self._optimize(self._find_argmax_location)
