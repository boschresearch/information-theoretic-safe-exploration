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

from copy import deepcopy
import gpytorch
import numpy as np
import scipy as sp
import torch

def point_is_within_box(point, box):
    '''
    Check whether or not point is inside given hyper-box.
    
    Parameters
    ----------
    point (torch.Tensor): point to be checked
    box (list of pairs of floats): list of the coordinates of the box's vertices

    Returns
    -------
    True if point is in box, False otherwise
    '''
    dimensions = len(box)
    point_is_inside_box = True
    for dimension in range(dimensions):
        point_coordinate = point[..., dimension]
        dimension_constraints = box[dimension]
        coordinate_is_inside_bounds = \
            (point_coordinate > dimension_constraints[0]) & (point_coordinate < dimension_constraints[1])
        point_is_inside_box = point_is_inside_box & coordinate_is_inside_bounds

    return point_is_inside_box


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


def sample_from_2d_gaussian(
        dimension, mean, short_dimension_variance, long_dimension_variance, number_of_samples, domain):
    '''
    Sample pairs of dimension-dimensional points [(x, z)], where the x points are distributed according to a bivariate 
    Gaussian with ellipsoid rtated of 45 degrees of center (mean, mean) and axis short_dimension_variance and 
    long_dimension_variance
    
    Parameters
    ----------
    dimension (int): dimensionality of the points of the pairs to sample
    mean (torch.Tensor): mean of the Gaussian distribution to sample from
    short_dimension_variance (float): short axis for the Gausian ellipsoid
    long_dimension_variance (float): long axis for the Gausian ellipsoid
    number_of_samples (int): number of samples to return
    domain: (list of pairs of floats): list of the coordinates of the domain vertices (domain is assumed to be a box)

    Returns
    -------
    Pair of tensors: one containing the first component of the sampled pairs, the other containing the second one
    '''

    theta = - np.pi / 4.
    rotation_matrix = torch.diag(torch.ones(2 * dimension)) * np.cos(theta)
    rotation_matrix[dimension:, :dimension] = torch.diag(torch.ones(dimension)) * np.sin(theta)
    rotation_matrix[:dimension, dimension:] = -torch.diag(torch.ones(dimension)) * np.sin(theta)

    unrotated_covariance = torch.diag(torch.ones(2 * dimension) * long_dimension_variance)
    unrotated_covariance[dimension:, dimension:] = torch.diag(torch.ones(dimension) * short_dimension_variance)

    rotated_covariance = rotation_matrix @ (unrotated_covariance @ rotation_matrix.T)

    augmented_mean = torch.cat((mean, mean), 1)

    distribution = torch.distributions.MultivariateNormal(loc=augmented_mean, covariance_matrix=rotated_covariance)
    samples = distribution.sample(sample_shape=torch.Size((2 * number_of_samples,)))
    samples_in_the_domain = samples[point_is_within_box(samples, domain)].unsqueeze(1)

    while len(samples_in_the_domain) < number_of_samples:
        new_samples = distribution.sample(sample_shape=torch.Size((number_of_samples,)))

        samples_in_the_domain = torch.cat(
            (samples_in_the_domain, new_samples[point_is_within_box(new_samples, domain)].unsqueeze(1)))

    return samples_in_the_domain[:, :, :dimension], samples_in_the_domain[:, :, dimension:]


def get_linearly_spaced_combinations(box, number_of_samples):
    """
        Return 2-D array with all linearly spaced combinations within the specified box.

        Parameters
        ----------
        box: (list of pairs of floats): list of the coordinates of the box's vertices
        num_samples: (int): number of samples to use for every dimension

        Returns
        -------
        combinations: A 2-d arrray. If d = len(box) and l = prod(number_of_samples) then it
            is of size l x d, that is, every row contains one combination of inputs.
        """

    dimensions = len(box)
    if dimensions == 1:
        return np.linspace(box[0][0], box[0][1], number_of_samples)[:, None]

    number_of_samples = [number_of_samples] * dimensions
    inputs = [np.linspace(box_bounds[0], box_bounds[1]) for box_bounds, samples in zip(box, number_of_samples)]

    return np.array([point.ravel() for point in np.meshgrid(*inputs)]).T


def sample_gp_function(gp_model, bounds, noise_variance):
    '''
    Return a function that interpolates a finite number of values sampled from the given zero-mean  prior

    Parameters
    ----------
    gp_model (gpytorch.models.ExactGP): GP model to sample from
    bounds (list of pairs of floats): list of the coordinates of the domain's vertices
    noise_variance (float): variance of the noise to add to the sampled function's evaluations
    mean_function (callable): GP prior mean function

    Returns
    -------
    Callable that interpolates finite number of samples from teh GP
    '''

    number_of_inputs_per_dimension = 60
    inputs = get_linearly_spaced_combinations(bounds, number_of_inputs_per_dimension)
    with gpytorch.settings.prior_mode(True):
        outputs = gp_model(
            torch.tensor(inputs, dtype=torch.float)).sample(sample_shape=torch.Size((1,))).squeeze().numpy()

    def gp_sample(x, noise=True):
        """
        Linear interpolator for GP sampled values

        Parameters
        ----------
        x (np.array): inputs for the function
        noise (bool): whether to include evaluation noise
        """
        x = np.atleast_2d(x)
        y = sp.interpolate.griddata(inputs, outputs, x, method='linear')

        # Work around weird dimension squishing in griddata
        y = np.atleast_2d(y.squeeze()).T

        if noise:
            noise_var = noise_variance(x) if callable(noise_variance) else noise_variance
            y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
        return torch.tensor(y, dtype=torch.float).squeeze(1)

    return gp_sample


def get_gp_sample_safe_at_origin(gp_prior, bounds, noise_variance, safety_threshold=0.):
    '''
    Sample an interpolator for a GP saple that is safe at the origin

    Parameters
    ----------
    gp_prior (gpytorch.models.ExactGP): GP model to sample from
    bounds (list of pairs of floats): list of the coordinates of the domain's vertices
    noise_variance (float): variance of the noise to add to the sampled function's evaluations
    safety_threshold (float): value that defines safety of parameters (i.e. x is safe if sample(x) >= 0)

    Returns
    -------
    Callable interpolator for a safe GP sample
    '''

    while True:
        test_gp = deepcopy(gp_prior)
        gp_sample = sample_gp_function(gp_prior, bounds, noise_variance)
        dimension = len(bounds)
        origin = torch.zeros((1, dimension))
        safety_bufer = 1
        upper_bound = 5
        value_at_origin = gp_sample(origin, False)
        noise = noise_variance(origin) if callable(noise_variance) else noise_variance
        test_gp.add_observations(origin, value_at_origin, torch.tensor([noise]))
        if test_gp.lower_confidence_bound(origin) > safety_threshold + safety_bufer and value_at_origin < upper_bound:
            return gp_sample


def sample_uniform_in_box(box, number_of_samples):
    '''
    Uiniformely sample points within given domain

    Parameters
    ----------
    box (list of pairs of floats): list of the coordinates of the vertices of the box it has to sample from
    number_of_samples (int) number of points to be sampled

    Returns
    -------
    number_of_samples samples from given box as torch.Tensor
    '''

    dimensions = len(box)
    box_sizes = torch.tensor([bounds[1] - bounds[0] for bounds in box], dtype=torch.float32)
    box_center = torch.tensor([box[i][0] + box_sizes[i] / 2. for i in range(len(box))], dtype=torch.float32)
    center_of_default_box = 0.5 * torch.ones(box_center.size())
    samples_in_origin_centered_box = torch.rand(number_of_samples, dimensions) - center_of_default_box

    return box_sizes * samples_in_origin_centered_box + box_center
