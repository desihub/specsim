# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage simulation configuration data.
"""
from __future__ import print_function, division

import os
import os.path
import math

import yaml

import numpy as np
import scipy.interpolate

import astropy.units


class Node(object):

    def __init__(self, value, path=[]):
        self.value = value
        self.path = path


    def __str__(self):
        return '.'.join(self.path)


    def get(self, path, required=True, default=None):
        """Get the value of a configuration parameter.
        """
        node = self.value
        node_path = self.path[:]
        for name in path.split('.'):
            node_path.append(name)
            if name not in node:
                if required:
                    raise KeyError(
                        'Missing required config node "{0}".'
                        .format('.'.join(node_path)))
                else:
                    return default
            node = node[name]
        return Node(node, node_path)


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
        self.name = self.get('name')

        # Initialize our wavelength grid.
        wave_config = self.get('wavelength')
        wave_min = wave_config.get('min').value
        wave_max = wave_config.get('max').value
        wave_step = wave_config.get('step').value
        nwave = 1 + int(math.floor((wave_max - wave_min) / wave_step))
        if nwave <= 0:
            raise ValueError('Invalid wavelength grid.')
        try:
            wave_unit = astropy.units.Unit(wave_config.get('unit').value)
        except exception as e:
            raise e
        self.wavelength = (wave_min + wave_step * np.arange(nwave)) * wave_unit

        # Use environment variables to interpolate {NAME} in the base path.
        try:
            self.base_path = self.get('base_path').value.format(**os.environ)
        except KeyError as e:
            raise ValueError('Environment variable not set: {0}.'.format(e))

        self.verbose = self.get('verbose')
        if self.verbose:
            print('Using config "{0}" with base path "{1}".'
                  .format(self.name, self.base_path))
            print(self.wavelength)


    def load_table(self, node, column_name, interpolate=True):
        """
        """
        if not node.path[-1] == 'table':
            raise ValueError(
                'Cannot load for non-table node "{0}"'.format(node))

        # Prepend our base path if this node's path is not already absolute.
        path = os.path.join(self.base_path, node.get('path').value)

        # Check that the required column names are present.
        required_names = [column_name]
        if interpolate:
            required_names.append('wavelength')
        required_names = sorted(required_names)

        columns = node.get('columns')
        config_column_names = sorted(columns.value.keys())
        if required_names != config_column_names:
            raise RuntimeError(
                'Expected names {0} for "{1}"'.format(required_names, columns))

        # Prepare the arguments we will send to astropy.table.Table.read()
        read_args = {}
        for key in ('format', 'hdu'):
            if key in node.value:
                read_args[key] = node.value[key]

        if self.verbose:
            print('Loading {0} with args {1}'.format(path, read_args))
        table = astropy.table.Table.read(path, **read_args)

        # Loop over columns to read.
        loaded_columns = {}
        for config_name in config_column_names:
            column = columns.get(config_name)
            # Look up the column data by index first, then by name.
            column_index = column.get('index', required=False)
            if column_index is not None:
                column_data = table.columns[column_index.value]
            else:
                column_data = table[column.get('name').value]
            # Resolve column units.
            column_unit = column.get('unit', required=False)
            if column_unit is not None:
                column_unit = astropy.units.Unit(column_unit.value)
            if column_data.unit is None:
                if column_unit is not None:
                    # Assign the unit specified in our config.
                    column_data.unit = column_unit
            else:
                if ((column_unit is not None) and
                    (column_unit != column_data.unit)):
                    raise RuntimeError(
                        'Units do not match for "{0}".'.format(column))

            loaded_columns[config_name] = column_data

        values = loaded_columns[column_name].data

        if interpolate:
            wavelength_column = loaded_columns['wavelength']
            # Convert wavelength column units if necesary.
            if wavelength_column.unit is None:
                raise RuntimeError(
                    'Wavelength units required for "{0}"'.format(columns))
            wavelength = wavelength_column.data * wavelength_column.unit
            if wavelength.unit != self.wavelength.unit:
                wavelength = wavelength.to(self.wavelength.unit)
            # Perform linear interpolation to our wavelength grid.
            interpolator = scipy.interpolate.interp1d(
                wavelength.value, values,
                kind='linear', copy=False, bounds_error=True, fill_value=None)
            values = interpolator(self.wavelength.value)

        # Apply units to the results if they are specified.
        values_unit = loaded_columns[column_name].unit
        if values_unit is not None:
            values = values * astropy.units.Unit(values_unit)

        return values


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
