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

import numpy as np
import torch

from ise.acquisitions.discrete_safe_acquisition_optimizer import DiscreteSafeAcquisitionOptimizer
from ise.utils.generic_utils import point_is_within_box

def safe_line_search_gradient_step(point, bounding_box, initial_learning_rate, safety_constraint=None):
    '''
    Performs a line-search gradient ascent step ensuring that the resulting point is within a given hyper-box
    and that it satisfy some given constraint
    
    Parameters
    ----------
    point (torch.Tensor): staring point for the gradient step
    bounding_box (list of pairs of floats): list of the coordinates of the box's vertices
    initial_learning_rate (float): learning rate for te gradient step
    safety_constraint (callable): optional additional contraint to be satisfied. It is consider satisfied if
    safety_constraint(end_point) >= 0.

    Returns
    -------
    End point after the gradient step and the used (possibly reduced) learning rate
    '''
    reduction_rate = 0.01
    number_of_tries = 50
    learning_rate = initial_learning_rate
    constraints_violated = torch.ones(point.shape[0], dtype=torch.bool)
    for i in range(number_of_tries):
        previous_value = point.detach().clone()
        point[constraints_violated] += (learning_rate * point.grad)[constraints_violated]
        constraints_violated = (torch.logical_not(point_is_within_box(point, bounding_box))).squeeze()
        if safety_constraint is not None:
            constraints_violated = constraints_violated | (safety_constraint(point) < 0).squeeze()

        if len(point[constraints_violated]) == 0:
            return point, learning_rate

        fraction_to_reduce_learning_rate_of = i * reduction_rate
        learning_rate[constraints_violated] *= 1 - fraction_to_reduce_learning_rate_of
        point[constraints_violated] = previous_value[constraints_violated]

    return point, learning_rate


class SimpleSafetyConstrainedOptimizer():
    def __init__(self, parameter, domain, learning_rate, safety_condition=None):
        self._parameter = parameter
        self._learning_rate = learning_rate * torch.ones(parameter.size())
        self._domain = domain
        self._safety_condition = safety_condition

    def step(self):
        '''
        Perform a gradient ascent step

        Returns
        -------
        Final learning rate used to perform the step without exiting the domain or violating the constraint
        '''

        with torch.no_grad():
            self._parameter, learning_rate = safe_line_search_gradient_step(
                self._parameter, self._domain, self._learning_rate, self._safety_condition)
            return learning_rate

    def zero_grad(self):
        '''
        Resets the gradients to zero
        '''

        self._parameter.grad.zero_()


class IseConstrainedOptimizer():
    def __init__(self, x, z, domain, initial_learning_rate, safety_condition):
        self.__x_optimizer = SimpleSafetyConstrainedOptimizer(
            x, domain, initial_learning_rate, safety_condition)
        self.__z = z

    def step(self):
        '''
        Perform a gradient ascent step

        Returns
        -------
        Final learning rate used to perform the step without exiting the domain or violating the constraint
        '''
        with torch.no_grad():
            learning_rate = self.__x_optimizer.step()
            self.__z += learning_rate * self.__z.grad

    def zero_grad(self):
        '''
        Resets the gradients to zero
        '''

        self.__x_optimizer.zero_grad()
        self.__z.grad.zero_()


def compute_safe_optimum(
        gp_model_objective, gp_model_constraint, domain, safe_seed, objective_function, constraint, variance_threshold):
    '''
    performs uncertainty sampling within the safe set until the variance over the safe set is smaller
    than 'variance_threshold'. At theat point it return the optimum of 'objective_function' within the safe set.

    Parameters
    ----------
    gp_model_objective (GaussianNoiseGP): GP model of the objective whose posterior is used to select sampling points
    gp_model_constraint (GaussianNoiseGP): GP model of the constraint whose posterior is used to define the safe set
    domain (list of int pairs): domain of the function
    safe_seed (torch.Tensor): safe seed
    objective_function (callable): funciton modeled by the GP
    variance_threshold (float): value of the variance at which to stop sampling

    Returns
    -------
    Value of safe optimum once the variance in the safe set is below 'variance_treshold'
    '''

    dimensions = len(domain)
    number_of_samples_per_dimension = 200
    number_of_samples = np.power(number_of_samples_per_dimension, dimensions)

    def uncertainty_sampling_acquisition(x):
        variance_objective = gp_model_objective.posterior(x).variance.squeeze()
        variance_constraint = gp_model_constraint.posterior(x).variance.squeeze()

        return torch.max(variance_objective, variance_constraint)

    discrete_variance_optimizer = DiscreteSafeAcquisitionOptimizer(
        gp_model_objective,
        safe_seed,
        uncertainty_sampling_acquisition,
        domain,
        number_of_samples,
        constraint_model=gp_model_constraint)

    max_variance = 100
    while max_variance > variance_threshold:
        next_point_to_sample, max_variance = discrete_variance_optimizer.optimize()
        gp_model_objective.add_observations(next_point_to_sample, objective_function(next_point_to_sample))
        gp_model_constraint.add_observations(next_point_to_sample, constraint(next_point_to_sample))

    discrete_objective_optimizer = DiscreteSafeAcquisitionOptimizer(
        gp_model_objective, safe_seed, objective_function, domain, number_of_samples, constraint_model=gp_model_constraint)

    return discrete_objective_optimizer.optimize()[1]
