# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage simulation configuration data.

Configuration data is normally loaded from a yaml file. Some standard
configurations are included with this package and can be loaded by name,
for example:

    >>> test_config = load_config('test')

Otherwise any filename with extension .yaml can be loaded::

    my_config = load_config('path/my_config.yaml')

Configuration data is accessed using attribute notation to specify a
sequence of keys:

    >>> test_config.name
    'Test Simulation'
    >>> test_config.atmosphere.airmass
    1.0

Use :meth:`Configuration.get_constants` to parse values with dimensions and
:meth:`Configuration.load_table` to load and interpolate tabular data.
"""
from __future__ import print_function, division

import os
import os.path
import math
import re
import warnings

import yaml

import numpy as np
import scipy.interpolate

import astropy.units
import astropy.table
import astropy.utils.data


class Node(object):
    """A single node of a configuration data structure.
    """
    def __init__(self, value, path=[]):
        self._assign('_value', value)
        self._assign('_path', path)


    def keys(self):
        return self._value.keys()


    def _assign(self, name, value):
        # Bypass our __setattr__
        super(Node, self).__setattr__(name, value)


    def __str__(self):
        return '.'.join(self._path)


    def __getattr__(self, name):
        # This method is only called when self.name fails.
        child_path = self._path[:]
        child_path.append(name)
        if name in self._value:
            child_value = self._value[name]
            if isinstance(child_value, dict):
                return Node(child_value, child_path)
            else:
                # Return the actual value for leaf nodes.
                return child_value
        else:
            raise AttributeError(
                'No such config node: {0}'.format('.'.join(child_path)))


    def __setattr__(self, name, value):
        # This method is always triggered by self.name = ...
        child_path = self._path[:]
        child_path.append(name)
        if name in self._value:
            child_value = self._value[name]
            if isinstance(child_value, dict):
                raise AttributeError(
                    'Cannot assign to non-leaf config node: {0}'
                    .format('.'.join(child_path)))
            else:
                self._value[name] = value
        else:
            raise AttributeError(
                'No such config node: {0}'.format('.'.join(child_path)))


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

    Attributes
    ----------
    wavelength : astropy.units.Quantity
        Array of linearly increasing wavelength values used for all simulation
        calculations.  Determined by the wavelength_grid configuration
        parameters.
    abs_base_path : str
        Absolute base path used for loading tabulated data.  Determined by
        the basepath configuration parameter.
    """
    def __init__(self, config):

        Node.__init__(self, config)
        self.update()


    def update(self):
        """Update this configuration.

        Updates the wavelength and abs_base_path attributes based on
        the current settings of the wavelength_grid and base_path nodes.
        """
        # Initialize our wavelength grid.
        grid = self.wavelength_grid
        nwave = 1 + int(math.floor(
            (grid.max - grid.min) / grid.step))
        if nwave <= 0:
            raise ValueError('Invalid wavelength grid.')
        wave_unit = astropy.units.Unit(grid.unit)
        wave = (grid.min + grid.step * np.arange(nwave)) * wave_unit
        self._assign('wavelength', wave)

        # Use environment variables to interpolate {NAME} in the base path.
        base_path = self.base_path
        if base_path == '<PACKAGE_DATA>':
            self._assign(
                'abs_base_path', astropy.utils.data._find_pkg_data_path('data'))
        else:
            try:
                self._assign('abs_base_path', base_path.format(**os.environ))
            except KeyError as e:
                raise ValueError('Environment variable not set: {0}.'.format(e))


    def get_constants(self, parent, required_names=None):
        """Interpret a constants node in this configuration.

        Parameters
        ----------
        parent : :class:`Node`
            Parent node in this configuration whose ``constants`` child
            will be processed.
        required_names : iterable or None
            List of constant names that are required to be present for this
            method to succeed.  If None, then no specific names are required.
            When specified, exactly these names are required and any other
            names will raise a RuntimeError.

        Returns
        -------
        dict
            Dictionary of (name, value) pairs where each value is an
            :class:`astropy.units.Quantity`.  When ``required_names`` is
            specified, they are guaranteed to be present as keys of the returned
            dictionary.

        Raises
        ------
        RuntimeError
            Constants present in the node do not match the required names.
        """
        constants = {}
        node = parent.constants
        names = sorted(node.keys())
        if required_names is not None and sorted(required_names) != names:
            raise RuntimeError(
                'Expected {0} for "{1}"'.format(required_names, node))
        for name in names:
            value = getattr(node, name)
            unit = None
            if isinstance(value, basestring):
                # A white space delimeter is required between value and units.
                tokens = value.split(None, 1)
                value = float(tokens[0])
                if len(tokens) > 1:
                    unit = tokens[1]
            constants[name] = astropy.units.Quantity(value, unit)
        return constants


    def load_table(self, parent, column_names, interpolate=True, as_dict=False):
        """Load and interpolate tabular data from one or more files.

        Reads a single file if parent.table.path exists, or else reads
        multiple files if parent.table.paths exists (and returns a dictionary).
        If as_dict is True, always return a dictionary using the 'default' key
        when only a single parent.table.path is present.
        """
        node = parent.table

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

        columns = node.columns
        config_column_names = sorted(columns.keys())
        if required_names != config_column_names:
            raise RuntimeError(
                'Expected {0} for "{1}"'.format(required_names, columns))

        # Prepare the arguments we will send to astropy.table.Table.read()
        read_args = {}
        keys = node.keys()
        for key in ('format', 'hdu'):
            if key in keys:
                read_args[key] = getattr(node, key)

        # Prepare a list of paths we will load tables from.
        paths = []
        path_keys = None
        try:
            # Look for parent.table.path first.
            paths.append(os.path.join(self.abs_base_path, node.path))
        except AttributeError:
            path_keys = node.paths.keys()
            for key in path_keys:
                path = getattr(node.paths, key)
                paths.append(os.path.join(self.abs_base_path, path))

        tables = {}
        # Loop over tables to load.
        for i, path in enumerate(paths):
            key = path_keys[i] if path_keys else 'default'

            with warnings.catch_warnings():
                warnings.simplefilter(
                    'ignore', category=astropy.units.core.UnitsWarning)
                table = astropy.table.Table.read(path, **read_args)

            if self.verbose:
                print('Loaded {0} rows from {1} with args {2}'
                      .format(len(table), path, read_args))

            # Loop over columns to read.
            loaded_columns = {}
            for config_name in config_column_names:
                column = getattr(columns, config_name)
                # Look up the column data by index first, then by name.
                try:
                    column_data = table.columns[column.index]
                except AttributeError:
                    column_data = table[column.name]
                column_values = column_data.data
                # Resolve column units.
                try:
                    column_unit = astropy.units.Unit(column.unit)
                except AttributeError:
                    column_unit = None
                try:
                    override_unit = column.override_unit
                    assert override_unit in (True, False)
                except AttributeError:
                    override_unit = False

                if override_unit or column_data.unit is None:
                    if column_unit is not None:
                        # Assign the unit specified in our config.
                        column_data.unit = column_unit
                else:
                    if ((column_unit is not None) and
                        (column_unit != column_data.unit)):
                        raise RuntimeError(
                            'Units do not match for "{0}".'.format(column))

                if interpolate:
                    loaded_columns[config_name] = column_data
                else:
                    unit = column_data.unit
                    if unit:
                        loaded_columns[config_name] = column_data.data * unit
                    else:
                        loaded_columns[config_name] = column_data.data

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
                try:
                    fill_value = node.extrapolated_value
                    bounds_error = False
                except AttributeError:
                    fill_value = None
                    bounds_error = True

                # Loop over other columns to interpolate onto our
                # wavelength grid.
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
                tables[key] = loaded_columns[column_names[0]]
            else:
                # Return a dictionary of all requested columns.
                tables[key] = loaded_columns

        if path_keys is None and not as_dict:
            return tables['default']
        else:
            return tables


def load_config(name, config_type=Configuration):
    """Load configuration data from a YAML file.

    Valid configuration files are YAML files containing no custom types, no
    sequences (lists), and with all mapping (dict) keys being valid python
    identifiers.

    Parameters
    ----------
    name : str
        Name of the configuration to load, which can either be a pre-defined
        name or else the name of a yaml file (with extension .yaml) to load.
        Pre-defined names are mapped to corresponding files in this package's
        data/config/ directory.

    Returns
    -------
    Configuration
        Initialized configuration object.

    Raises
    ------
    ValueError
        File name has wrong extension or does not exist.
    RuntimeError
        Configuration data failed a validation test.
    """
    base_name, extension = os.path.splitext(name)
    if extension not in ('', '.yaml'):
        raise ValueError('Config file must have .yaml extension.')
    if extension:
        file_name = name
    else:
        file_name = astropy.utils.data._find_pkg_data_path(
            'data/config/{0}.yaml'.format(name))
    if not os.path.isfile(file_name):
        raise ValueError('No such config file "{0}".'.format(file_name))

    # Validate that all mapping keys are valid python identifiers.
    valid_key = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*\Z')
    with open(file_name) as f:
        next_value_is_key = False
        for token in yaml.scan(f):
            if isinstance(
                token,
                (yaml.BlockSequenceStartToken, yaml.FlowSequenceStartToken)):
                raise RuntimeError('Config sequences not implemented yet.')
            if next_value_is_key:
                if not isinstance(token, yaml.ScalarToken):
                    raise RuntimeError(
                        'Invalid config key type: {0}'.format(token))
                if not valid_key.match(token.value):
                    raise RuntimeError(
                        'Invalid config key name: {0}'.format(token.value))
            next_value_is_key = isinstance(token, yaml.KeyToken)

    with open(file_name) as f:
        return config_type(yaml.safe_load(f))
