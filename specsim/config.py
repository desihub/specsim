# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage simulation configuration data.
"""
from __future__ import print_function, division

import os
import os.path

import yaml


class Configuration(object):
    """Configuration parameters container and utilities.

    This class specifies the required top-level keys and delegates the
    interpretation and validation of their values to other functions.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters, normally obtained by parsing
        a YAML file with :func:`load`.
    verbose : bool
        Print verbose details while performing this operation.

    Raises
    ------
    ValueError
        Missing required top-level configuration key.
    """
    def __init__(self, config, verbose=True):

        self.config = config

        for required in ('name', 'base_path', 'atmosphere'):
            if required not in self.config:
                raise ValueError(
                    'Missing required config key: {0}.'.format(required))

        # Use environment variables to interpolate {NAME} in the base path.
        try:
            self.base_path = self.config['base_path'].format(**os.environ)
        except KeyError as e:
            raise ValueError('Environment variable not set: {0}.'.format(e))

        if verbose:
            print('Using config "{0}" with base path "{1}".'
                  .format(config['name'], self.base_path))
            print(config)


def load(filename, verbose):
    """Load configuration data from a YAML file.

    Parameters
    ----------
    filename : str
        Name of a YAML file to read.
    verbose : bool
        Print verbose details while performing this operation.

    Returns
    -------
    Configuration
        Initialized configuration object.
    """
    with open(filename) as f:
        return Configuration(yaml.safe_load(f), verbose=verbose)
