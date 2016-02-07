# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage simulation configuration data.
"""
from __future__ import print_function, division

import os
import os.path

import yaml

import specsim.spectrum


class Node(object):

    def __init__(self, value, path=''):
        self.value = value
        self.path = path


    def __str__(self):
        return 'Node({},{})'.format(self.path, type(self.value))


    def get(self, path, required=True, default=None):
        """Get the value of a configuration parameter.
        """
        node = self.value
        names = path.split('.')
        for depth, name in enumerate(names):
            if name not in node:
                if required:
                    raise KeyError(
                        'Missing required config node "{0}".'
                        .format('.'.join(names[:depth+1])))
                else:
                    return default
            node = node[name]
        full_path = self.path + '.' + path if self.path else path
        return Node(node, full_path)


class Configuration(Node):
    """Configuration parameters container and utilities.

    This class specifies the required top-level keys and delegates the
    interpretation and validation of their values to other functions.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters, normally obtained by parsing
        a YAML file with :func:`load`.

    Raises
    ------
    ValueError
        Missing required top-level configuration key.
    """
    def __init__(self, config):

        Node.__init__(self, config)

        # Check for required top-level nodes.
        for required in ('name', 'base_path', 'verbose', 'atmosphere'):
            self.get(required)

        # Use environment variables to interpolate {NAME} in the base path.
        try:
            self.base_path = self.get('base_path').value.format(**os.environ)
        except KeyError as e:
            raise ValueError('Environment variable not set: {0}.'.format(e))

        self.verbose = self.get('verbose')
        if self.verbose:
            print('Using config "{0}" with base path "{1}".'
                  .format(config['name'], self.base_path))


    def load_table(self, node):
        """
        """
        if not node.path.endswith('.table'):
            raise ValueError(
                'Cannot load for non-table node {0}.'.format(node.path))
        # Prepend our base path if this node's path is not already absolute.
        path = os.path.join(self.base_path, node.value)
        return specsim.spectrum.SpectralFluxDensity.load(path)


def load(filename, verbose):
    """Load configuration data from a YAML file.

    Parameters
    ----------
    filename : str
        Name of a YAML file to read.

    Returns
    -------
    Configuration
        Initialized configuration object.
    """
    with open(filename) as f:
        return Configuration(yaml.safe_load(f))
