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

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

from ise.acquisitions.line_bo_acquisiton_base import LineBoAcquisitionBase
from ise.acquisitions.discrete_safe_acquisition_optimizer import DiscreteSafeAcquisitionOptimizer
from ise.utils.generic_utils import sample_uniform_in_box, point_is_within_box

class DummyGPModelAlwaysSafe:
    def lower_confidence_bound(self, x):
        return torch.ones(x.shape[0], 1, 1)

class MesLineBoAcquisition(LineBoAcquisitionBase):
    def __init__(self, gp_model_objective, safe_seed, domain, mes_candidate_size=50, gp_model_safety=None):
        '''
        Constructor

        Parameters
        ----------
        gp_model_objective (gpytorch.models.ExactGP): GP model of the objective function, whose posterior
        is used by the MES acquisition function
        safe_seed (torch.Tensor): initial safe evaluation parameter
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        mes_candidates_number (int): Number of points of the candidate set for the MES acquisition
        gp_model_safety (gpytorch.models.ExactGP): GP model of the safety constraint, used to compute safe set if set.
        If None, then the acquisition function is unconstrained and can select any parameter in the domain
        '''

        super().__init__(safe_seed, domain)
        is_constrained = gp_model_safety is not None
        self._model_objective = gp_model_objective
        domain_as_tensor = torch.tensor(domain, dtype=torch.float32)
        candidate_set = torch.rand(mes_candidate_size, len(domain), dtype=domain_as_tensor.dtype)
        candidate_set = domain_as_tensor[0, 0] + \
                        (domain_as_tensor[0, 1] - domain_as_tensor[0, 0]) * candidate_set
        mes_acquisition = qMaxValueEntropy(gp_model_objective, candidate_set)
        if is_constrained:
            self._acquisition = DiscreteSafeAcquisitionOptimizer(
                gp_model_objective, safe_seed, mes_acquisition, domain, None, gp_model_safety)
        else:
            self._acquisition = DiscreteSafeAcquisitionOptimizer(
                gp_model_objective, safe_seed, mes_acquisition, domain, None, DummyGPModelAlwaysSafe())


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
        (torch.Tensor) Point that maximizes acquisition function along given line
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

        return self._acquisition.optimize(rembedded_samples)


    def optimize(self):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain

        Returns
        -------
        (torch.Tensor) Point that maximizes acquisition function on a sample of random 1d subsets
        '''

        return self._optimize(self._find_argmax_location)
