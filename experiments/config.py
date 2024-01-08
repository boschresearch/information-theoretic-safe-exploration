# Copyright (C) Robert Bosch GmbH 2020.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
# This program and the accompanying materials are made available under
# the terms of the Bosch Internal Open Source License v4 and later
# which accompanies this distribution, and is available at
# http://bios.intranet.bosch.com/bioslv4.txt

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