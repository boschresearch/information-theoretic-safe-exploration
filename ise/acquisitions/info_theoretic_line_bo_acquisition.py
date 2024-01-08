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

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
import math
import torch

from ise.acquisitions.line_bo_acquisiton_base import LineBoAcquisitionBase
from ise.acquisitions.ise_acquisition import IseAcquisition
from ise.utils import generic_utils, sample_uniform_in_box

class InfoTheoreticLineBoAcquisition(LineBoAcquisitionBase):
    def __init__(
            self, gp_model_safety, gp_model_ojective, safe_seed, domain, mes_candidates_number, noise_variance=None):
        '''
        Constructor

        Parameters
        ----------
        gp_model_safety (gpytorch.models.ExactGP): GP that models the safety constraint function
        gp_model_objective (gpytorch.models.ExactGP): GP that models the objective function
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        mes_candidates_number (int): Number of points of the candidate set for the MES acquisition
        noise_variance (callable or None): Optional, to be provided in case of heteroskedastic noise it maps input
        locations to the corresponding observation noise
        '''

        super().__init__(safe_seed, domain)
        self._model_safety = gp_model_safety
        self._relevant_lengthscale_safe_model = gp_model_safety.covar_module.base_kernel.lengthscale.item()
        self._ise_acquisition = IseAcquisition(
            gp_model_safety, safe_seed, domain, 0, 0, 0, noise_variance)

        candidate_set = torch.rand(mes_candidates_number, len(domain), dtype=torch.float32)
        transposed_domain = torch.tensor(domain, dtype=torch.float32).T
        candidate_set = transposed_domain[0] + (transposed_domain[1] - transposed_domain[0]) * candidate_set
        self._mes_acquisition = qMaxValueEntropy(gp_model_ojective, candidate_set)


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
        subspace_augmented_samples_ise_x, subspace_augmented_samples_ise_z = generic_utils.sample_from_2d_gaussian(
            subspace_dimensions,
            gaussian_mean,
            self._relevant_lengthscale_safe_model,
            subspace_length,
            number_of_samples,
            standard_subspace_bounds
        )
        subspace_augmented_mes_samples = sample_uniform_in_box(standard_subspace_bounds, number_of_samples)
        rembedded_samples_ise_x = self._one_d_samples_to_full_domain(
            subspace_augmented_samples_ise_x, origin, normalized_direction)
        rembedded_samples_ise_z = self._one_d_samples_to_full_domain(
            subspace_augmented_samples_ise_z, origin, normalized_direction)
        rembedded_samples_mes = self._one_d_samples_to_full_domain(
            subspace_augmented_mes_samples, origin, normalized_direction)

        ise_samples_are_in_domain = generic_utils.point_is_within_box(rembedded_samples_ise_x, self._domain)
        mes_samples_are_in_domain = generic_utils.point_is_within_box(rembedded_samples_mes, self._domain)
        rembedded_samples_ise_x = rembedded_samples_ise_x[ise_samples_are_in_domain]
        rembedded_samples_ise_z = rembedded_samples_ise_z[ise_samples_are_in_domain]
        rembedded_samples_mes = rembedded_samples_mes[mes_samples_are_in_domain]
        if len(rembedded_samples_ise_x) == 0 or len(rembedded_samples_mes) == 0:
            return None, None

        ise_samples_are_safe = (self._model_safety.lower_confidence_bound(rembedded_samples_ise_x) >=0).squeeze()
        mes_samples_are_safe = (self._model_safety.lower_confidence_bound(rembedded_samples_mes) >=0).squeeze()
        rembedded_samples_ise_x = rembedded_samples_ise_x[ise_samples_are_safe]
        rembedded_samples_ise_z = rembedded_samples_ise_z[ise_samples_are_safe]
        rembedded_samples_mes = rembedded_samples_mes[mes_samples_are_safe]
        if len(rembedded_samples_ise_x) == 0 or len(rembedded_samples_mes) == 0:
            return None, None

        number_of_samples_to_compute_in_parallel = 500
        number_of_batches = math.ceil(rembedded_samples_ise_x.shape[1] / number_of_samples_to_compute_in_parallel)
        ise_values = torch.empty(0)
        mes_values = torch.empty(0)
        for i in range(number_of_batches):
            start_index = i * number_of_samples_to_compute_in_parallel
            end_index = start_index + number_of_samples_to_compute_in_parallel if i < number_of_batches - 1 else None
            ise_values_of_current_batch = self._ise_acquisition.compute_acquisition_value(
                rembedded_samples_ise_x[start_index: end_index].unsqueeze(1),
                rembedded_samples_ise_z[start_index: end_index].unsqueeze(1))
            rembedded_samples_mes_current_batch = rembedded_samples_mes[start_index: end_index]
            mes_values_of_current_batch = self._mes_acquisition(rembedded_samples_mes_current_batch
                .view(len(rembedded_samples_mes_current_batch), 1, rembedded_samples_mes.shape[1])
            )
            if ise_values_of_current_batch.dim() == 0:
                ise_values_of_current_batch.unsqueeze(0)
            if mes_values_of_current_batch.dim() == 0:
                mes_values_of_current_batch.unsqueeze(0)
            ise_values = torch.cat((ise_values, ise_values_of_current_batch))
            mes_values = torch.cat((mes_values, mes_values_of_current_batch))

        ise_max_value, ise_max_values_index = torch.topk(ise_values, 1)
        mes_max_value, mes_max_values_index = torch.topk(mes_values, 1)

        if ise_max_value > mes_max_value:
            return rembedded_samples_ise_x[ise_max_values_index].squeeze(0), ise_max_value

        return rembedded_samples_mes[mes_max_values_index], mes_max_value


    def optimize(self):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain

        Returns
        -------
        (torch.Tensor) Point that maximizes acquisition function on a sample of random 1d subsets
        '''

        return self._optimize(self._find_argmax_location)
