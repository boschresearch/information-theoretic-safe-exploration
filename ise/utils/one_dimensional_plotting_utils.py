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
from matplotlib import pyplot as plt
import torch

class OneDGPPlotInfos():
    def __init__(
            self,
            gp_model,
            true_function,
            plot_interval,
            total_safe_set,
            current_safe_set,
            next_evaluation_point,
            confidence):
        '''
        Constructor

        Parameters
        ----------
        gp_model (gpytorch.models.ExactGP): GP model object
        true_function (callable): Latent function modeled with the GP
        plot_interval (pair of floats): Extremes of the interval within which the functions has to be plotted
        total_safe_set (list of pair of floats [[a, b], ]): Safe set of the true function
        current_safe_set (list of pair of floats): Safe set according to the current posterior
        next_evaluation_point (float): Next point to be evaluated
        confidence (float): width of the confidence interval in units of the posterior standard deviation
        '''

        self.gp_model = gp_model
        self.true_function = true_function
        self.plot_interval = plot_interval
        self.total_safe_set = total_safe_set
        self.current_safe_set = current_safe_set
        self.next_evaluation_point = next_evaluation_point
        self.confidence = confidence


def _plot_x_axis_interval(plot_axes, interval, color, y_coordinate):
    '''
    Adds an horizontal line in the provided interval at y_coordinate

    Parameters
    ----------
    plot_axes: Plot to add the line to
    interval (pair of float [a, b]): Extremes of the interval
    color (string): Color of the line
    y_coordinate (float): y coordinate at which the line should appear
    '''

    if not interval:
        return
    x_coordinates = torch.linspace(interval[0], interval[1], 2)
    y_coordinates = torch.tensor([y_coordinate] * 2)
    plot_axes.plot(x_coordinates, y_coordinates, color=color, linewidth=12)


def _plot_multiple_x_axis_intervals(plot_axes, intervals, color, y_coordinate):
    '''
    Adds an horizontal line in the provided multiple intervals at y_coordinate

    Parameters
    ----------
    plot_axes: Plot to add the line to
    interval (list of pair of float [[a, b], ]): List of extremes of the intervals
    color (string): Color of the line
    y_coordinate (float): y coordinate at which the lines should appear
    '''

    for interval in intervals:
        _plot_x_axis_interval(plot_axes, interval, color, y_coordinate)


def _plot_safe_sets(total_safe_set, current_safe_set, plot_axes):
    '''
    Plots total and current safe sets in plot_axes, at the bottom of the figure

    Parameters
    ----------
    total_safe_set (list of intervals extremes of the true safe set): Complete true safe set
    current_safe_set (list of intervals extremes of the current safe set): Safe set according to the latest posterior
    plot_axes: Plot to add the lines to
    '''

    _plot_multiple_x_axis_intervals(plot_axes, total_safe_set, 'deepskyblue', plot_axes.get_ylim()[0])
    _plot_multiple_x_axis_intervals(plot_axes, current_safe_set, 'aqua', plot_axes.get_ylim()[0])


def plot_data_and_posterior(gp_plot_info):
    '''
    Plots the mean and confidence interval of the posterior GP, together with the observations

    Parameters
    ----------
    gp_plot_info (OneDGPPlotInfos object): Object containing info about the plot and objects to plot
    '''

    data = gp_plot_info.gp_model.evaluated_points
    plot_interval_start = gp_plot_info.plot_interval[0]
    plot_interval_end = gp_plot_info.plot_interval[1]
    if data.numel() != 0:
        if torch.min(data) < plot_interval_start or torch.max(data) > plot_interval_end:
            raise ValueError("Given interval does not include observed data!")

    with torch.no_grad():
        interval_length = plot_interval_end - plot_interval_start
        points_per_length_unit = 50
        sample_points = torch.linspace(
            plot_interval_start, plot_interval_end, math.ceil(interval_length * points_per_length_unit))
        sample_predicitons = gp_plot_info.gp_model(sample_points)
        posterior_mean = sample_predicitons.mean
        scaled_confidence = sample_predicitons.stddev.mul_(gp_plot_info.confidence)

        axes = plt.subplots(1, 1, figsize=(20, 10))[1]
        scatter_data = axes.plot(data.numpy(), gp_plot_info.gp_model.observed_values.numpy(), 'x')[0]
        mean_plot = axes.plot(sample_points.numpy(), posterior_mean.numpy(), 'b')[0]
        confidence_interval_area = axes.fill_between(
            sample_points.numpy(),
            posterior_mean.sub(scaled_confidence).numpy(),
            posterior_mean.add(scaled_confidence).numpy(),
            alpha=0.3)
        plot_legend = (
            [scatter_data, mean_plot, confidence_interval_area], ['Observed Data', 'Mean', 'Confidence Interval'])

        latent_function_plot = axes.plot(sample_points, gp_plot_info.true_function(sample_points), 'r')[0]
        plot_legend[0].append(latent_function_plot)
        plot_legend[1].append('True function')

        min_y_value, max_y_value = axes.get_ylim()
        axes.vlines(gp_plot_info.next_evaluation_point, min_y_value, max_y_value, colors='g', linestyles='dashed')

        _plot_safe_sets(gp_plot_info.total_safe_set, gp_plot_info.current_safe_set, axes)

        axes.legend(plot_legend[0], plot_legend[1])


def get_1d_safe_set(function, function_legth_scale, interval):
    '''
    Calculates set where the function is bigger than 0, in the give ninterval

    Parameters
    ----------
    function (callable): function whose safe set has to be evaluated
    function_legth_scale (float): length scale of the function, i.e. typical length within which the function
    changes significantly
    interval (pair of floats  (a, b)): Extremes of the interval within which the function willl be evaluated

    Returns
    -------
    Safe set as list of intervals ([[a, b], [a, b], ...])
    '''

    number_of_sample_points_per_bin = 100
    interval_start = interval[0]
    interval_end = interval[1]
    interval_length = interval_end - interval_start
    number_of_bins = math.ceil(interval_length / function_legth_scale)

    sample_points_x = torch.linspace(interval_start, interval_end, number_of_bins * number_of_sample_points_per_bin)
    sample_points_y = function(sample_points_x)

    currently_inside_a_safe_interval = False
    safe_intervals = []
    current_safe_interval = 0

    for i in range(number_of_bins * number_of_sample_points_per_bin):
        if sample_points_y[i].item() > 0:
            if not currently_inside_a_safe_interval:
                currently_inside_a_safe_interval = True
                current_point = sample_points_x[i].item()
                safe_intervals.append([current_point, current_point])
            else:
                safe_intervals[current_safe_interval][1] = sample_points_x[i].item()
        else:
            if currently_inside_a_safe_interval:
                current_safe_interval += 1
                currently_inside_a_safe_interval = False

    return safe_intervals
