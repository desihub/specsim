# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an astronomical source for spectroscopic simulations.
"""
from __future__ import print_function, division

import numpy as np

import scipy.interpolate

import astropy.units as u


class Source(object):
    """
    """
    def __init__(self, name, type_name, wavelength, flux):
        self.name = name
        self.type_name = type_name
        self.wavelength = wavelength
        self.flux = flux


    def update(self, type_name, wavelength, flux, extrapolate_value=None):
        """
        """
        self.type_name = type_name
        # Convert input arrays to original units.
        try:
            wavelength_value = wavelength.to(self.wavelength.unit).value
        except AttributeError:
            wavelength_value = np.asarray(wavelength)
        try:
            flux_value = flux.to(self.flux.unit).value
        except AttributeError:
            flux_value = np.asarray(flux)

        # Interpolate to the simulation wavelengths if necessary.
        if not np.array_equal(wavelength_value, self.wavelength.value):
            bounds_error = extrapolate_value is None
            interpolator = scipy.interpolate.interp1d(
                wavelength_value, flux_value,
                kind='linear', copy=False, assume_sorted=True,
                bounds_error=bounds_error, fill_value=extrapolate_value)
            flux_value = interpolator(self.wavelength.value)

        self.flux = flux_value * self.flux.unit


def initialize(config):
    """Initialize the source model from configuration parameters.

    Parameters
    ----------
    config : :class:`specsim.config.Configuration`
        The configuration parameters to use.

    Returns
    -------
    Source
        An initialized source model.
    """
    # Check for required top-level config nodes.
    flux = config.load_table(config.source, 'flux')
    return Source(
        config.source.name, config.source.type, config.wavelength, flux)
