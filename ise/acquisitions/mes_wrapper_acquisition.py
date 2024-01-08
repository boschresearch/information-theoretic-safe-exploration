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
from botorch.optim.optimize import optimize_acqf
import torch

class MesAcquisitionWrapper():
    def __init__(self, gp_model, domain):
        '''
        Constructor

        Parameters
        ----------
        gp_model (gpytorch.models.ExactGP): GP that models the objective function
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        '''

        self._domain = domain
        candidate_set = torch.rand(50, len(domain), dtype=torch.tensor(domain).dtype)
        transposed_domain = torch.tensor(domain).T
        candidate_set = transposed_domain[0] + (transposed_domain[1] - transposed_domain[0]) * candidate_set
        self._mes_acquisition = qMaxValueEntropy(gp_model, candidate_set)

    def optimize(self):
        '''
        Finds point that maximizes acquisition function

        Returns
        -------
        (torch.Tensor) Point that maximizes acquisition function in the domain
        '''
        return optimize_acqf(
            acq_function = self._mes_acquisition,
            bounds = torch.tensor(self._domain).T,
            q = 1,
            num_restarts = 30,
            raw_samples = 500,
        )