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

from ise.utils.generic_utils import sample_uniform_in_box

class DiscreteSafeAcquisitionOptimizer:
    def __init__(self, gp_model, safe_seed, acquisition_function, domain, sample_size, constraint_model=None):
        '''
        Constructor

        Parameters
        ----------
        gp_model (GaussianNoiseGP): GP model whose posterior is used by the acquisition function
        safe_seed (torch.Tensor): safe seed for the safe optimization
        acquisition_function: callable object that computes acquitition values
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        sample_size: (int) number of points to sample in the domain when searching
        for the argmax of the acquisition
        constraint_model (GaussianNoiseGP): GP that models the safety constraint, if different from the objective
        '''

        self._model = gp_model
        self._constraint_model = constraint_model if constraint_model is not None else gp_model
        self._safe_seed = safe_seed
        self._domain = domain
        self._relevant_lengthscale = gp_model.covar_module.base_kernel.lengthscale.item()
        self._acquisition_function = acquisition_function
        self._sample_size = sample_size


    def optimize(self, evaluation_points=None):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain

        Parameters
        ----------
        evaluation_points (torch.Tensor): candidate points at which to evaluate the acquisition. If None they will
        be samples randomly from the domain

        Returns
        -------
        (torch.Tensor) A point in the safe set that achieves the highest acquisition value among the smapled points
        '''

        if evaluation_points is None:
            sample_points = sample_uniform_in_box(self._domain, self._sample_size).unsqueeze(1)
        else:
            sample_points = evaluation_points
            self._sample_size = len(sample_points)

        number_of_samples_to_compute_in_parallel = 1000
        number_of_batches = math.ceil(self._sample_size / number_of_samples_to_compute_in_parallel)
        max_value = -1e6
        index_of_max_value = -1e6
        with torch.no_grad():
            for i in range(number_of_batches):
                start_index = i * number_of_samples_to_compute_in_parallel
                end_index = \
                    start_index + number_of_samples_to_compute_in_parallel if i < number_of_batches - 1 else None
                current_batch_points = sample_points[start_index: end_index]
                points_are_safe = self._constraint_model.lower_confidence_bound(current_batch_points) >= 0
                safe_indices = torch.nonzero(points_are_safe.view(points_are_safe.shape[0])).squeeze() + start_index
                if safe_indices.nelement() == 0:
                    continue
                elif len(safe_indices.size()) == 0:
                    safe_indices = safe_indices.unsqueeze(0)

                values_of_current_safe_batch = self._acquisition_function(
                    current_batch_points[points_are_safe.squeeze()].view(
                        len(safe_indices), *current_batch_points.shape[1:]))
                current_batch_max, current_batch_argmax = torch.topk(values_of_current_safe_batch, 1)
                if current_batch_max > max_value:
                    max_value = current_batch_max
                    index_of_max_value = safe_indices[current_batch_argmax]

        argmax = sample_points[index_of_max_value].view(self._safe_seed.shape) if index_of_max_value >= 0 \
            else self._safe_seed

        return argmax, max_value
