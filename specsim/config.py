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
import astropy.table


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
        self.name = self.get('name').value

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

        # Define custom dimensionless units used for configuration.
        self.row_unit = astropy.units.def_unit(
            ['row'], astropy.units.Unit(1), doc='row: one CCD row',
            namespace=vars(astropy.units.astrophys))

        self.verbose = self.get('verbose').value
        if self.verbose:
            print('Using config "{0}" with base path "{1}".'
                  .format(self.name, self.base_path))


    def get_constants(self, parent, required_names=None):
        """
        """
        constants = {}
        node = parent.get('constants')
        names = sorted(node.value.keys())
        if required_names is not None and sorted(required_names) != names:
            raise RuntimeError(
                'Expected {0} for "{1}"'.format(required_names, node))
        for name in names:
            value = node.get(name).value
            unit = None
            if isinstance(value, basestring):
                # A white space delimeter is required between value and units.
                tokens = value.split(None, 1)
                value = float(tokens[0])
                if len(tokens) > 1:
                    unit = tokens[1]
            constants[name] = astropy.units.Quantity(value, unit)
        return constants


    def load_table(self, parent, column_names, interpolate=True):
        """
        """
        node = parent.get('table')

        # Prepend our base path if this node's path is not already absolute.
        path = os.path.join(self.base_path, node.get('path').value)

        # Check that the required column names are present.
        if isinstance(column_names, basestring):
            return_scalar = True
            column_names = [column_names]
        else:
            return_scalar = False

        required_names = column_names[:]
        if interpolate:
            required_names.append('wavelength')
        required_names = sorted(required_names)

        columns = node.get('columns')
        config_column_names = sorted(columns.value.keys())
        if required_names != config_column_names:
            raise RuntimeError(
                'Expected {0} for "{1}"'.format(required_names, columns))

        # Prepare the arguments we will send to astropy.table.Table.read()
        read_args = {}
        for key in ('format', 'hdu'):
            if key in node.value:
                read_args[key] = node.value[key]

        table = astropy.table.Table.read(path, **read_args)
        if self.verbose:
            print('Loaded {0} rows from {1} with args {2}'
                  .format(len(table), path, read_args))

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
            column_values = column_data.data
            # Resolve column units.
            column_unit = column.get('unit', required=False)
            if column_unit is not None:
                column_unit = astropy.units.Unit(column_unit.value)
            override_unit = column.get(
                'override_unit', required=False, default=False)
            if override_unit:
                override_unit = override_unit.value # boolean

            if override_unit or column_data.unit is None:
                if column_unit is not None:
                    # Assign the unit specified in our config.
                    column_data.unit = column_unit
            else:
                if ((column_unit is not None) and
                    (column_unit != column_data.unit)):
                    raise RuntimeError(
                        'Units do not match for "{0}".'.format(column))

            loaded_columns[config_name] = column_data

        if interpolate:
            wavelength_column = loaded_columns['wavelength']
            # Convert wavelength column units if necesary.
            if wavelength_column.unit is None:
                raise RuntimeError(
                    'Wavelength units required for "{0}"'.format(columns))
            wavelength = wavelength_column.data * wavelength_column.unit
            if wavelength.unit != self.wavelength.unit:
                wavelength = wavelength.to(self.wavelength.unit)

            # Initialize extrapolation if requested.
            fill_value = node.get('extrapolated_value', required=False)
            if fill_value is None:
                bounds_error = True
            else:
                fill_value = fill_value.value
                bounds_error = False

            # Loop over other columns to interpolate onto our wavelength grid.
            for column_name in column_names:
                interpolator = scipy.interpolate.interp1d(
                    wavelength.value, loaded_columns[column_name].data,
                    kind='linear', copy=False,
                    bounds_error=bounds_error, fill_value=fill_value)
                interpolated_values = interpolator(self.wavelength.value)
                unit = loaded_columns[column_name].unit
                if unit:
                    interpolated_values = interpolated_values * unit
                loaded_columns[column_name] = interpolated_values

            # Delete the temporary wavelength column now we have
            # finished using it for interpolation.
            del loaded_columns['wavelength']

        if return_scalar:
            # Return just the one column that was requested.
            return loaded_columns[column_names[0]]
        else:
            # Return a dictionary of all requested columns.
            return loaded_columns


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
