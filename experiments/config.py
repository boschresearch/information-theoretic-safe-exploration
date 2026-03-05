# Copyright (c) 2023 Robert Bosch GmbH
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

"""Contains the default configurations for our experiments.

Note that this doesn't have to be a Python file, but could also be a static file. For
example, a .toml or  ha .json file.
"""

default_one_d_exp_parameters = dict(
    learning_rate=0,
    epochs=0,
    beta=4,
    domain=[(-2.4, 10.5)],
    random_seed=3403,
    method='safe_opt',
    noise_variance=0.05,
    kernel_lengthscale=0.6,
    lipschitz_constant=1
)

default_gp_samples_exp_parameters = dict(
    learning_rate=0,
    epochs=0,
    beta=4,
    domain=[(-1., 1.), (-1., 1.)],
    random_seed=7,
    method='safe_opt',
    noise_variance=0.05,
    kernel_lengthscale=0.3,
    lipschitz_constant=1,
    output_scale=30,
    mes_candidate_set_size=50,
    objective_is_constraint=False
)

default_pd_controller_exp_parameters = dict(
    learning_rate=0,
    epochs=0,
    random_seed=7,
    method='safe_opt',
    noise_variance=0.05,
    lipschitz_constant=1
)

default_parameters_line_bo = dict(
    beta=4,
    dimensions=4,
    random_seed=35,
    lipschitz_constant=1,
    scaled_domain=True,
    method='safe_opt',
    high_noise=0.8,
)
