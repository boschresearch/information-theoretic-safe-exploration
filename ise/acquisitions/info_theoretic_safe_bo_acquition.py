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
import torch

from ise.acquisitions.discrete_safe_acquisition_optimizer import DiscreteSafeAcquisitionOptimizer
from ise.acquisitions.ise_acquisition import IseAcquisition

class InfoTheoreticSafeBoAcquisition():
    def __init__(
            self,
            gp_model_safety,
            gp_model_ojective,
            domain,
            safe_seed,
            number_of_samples,
            learning_rate,
            learning_epochs,
            mes_candidates):
        '''
        Constructor

        Parameters
        ----------
        gp_model_safety (gpytorch.models.ExactGP): GP that models the safety constraint function
        gp_model_objective (gpytorch.models.ExactGP): GP that models the objective function
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        safe_seed (torch.Tensor): initial safe seed
        number_of_samples (int): number of samples to use to find optimization starting points for the ISE acquisition
        learning_rate (double): learning rate for the ISE acquisition
        learning_epochs (int): number of epochs for the ISE acquisition
        mes_candidates_number (int): Number of points of the candidate set for the MES acquisition
        '''

        self._ise_acquisition = IseAcquisition(
            gp_model_safety, safe_seed, domain, learning_rate, learning_epochs, number_of_samples)
        candidate_set = torch.rand(mes_candidates, len(domain), dtype=torch.tensor(domain).dtype)
        transposed_domain = torch.tensor(domain).T
        candidate_set = transposed_domain[0] + (transposed_domain[1] - transposed_domain[0]) * candidate_set

        self._mes_acquisition = DiscreteSafeAcquisitionOptimizer(
            gp_model_ojective,
            safe_seed,
            qMaxValueEntropy(gp_model_ojective, candidate_set),
            domain,
            number_of_samples,
            gp_model_safety)

    def optimize(self):
        '''
        Finds points that maximizes acquisition function

        Returns
        -------
        (torch.Tensor) Point that maximises acquisition
        '''
        ise_argmax, ise_max = self._ise_acquisition.optimize()
        mes_argmax, mes_max = self._mes_acquisition.optimize()

        if ise_max >= mes_max:
            return ise_argmax, ise_max

        return mes_argmax, mes_max