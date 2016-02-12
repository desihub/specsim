# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an astronomical source for spectroscopic simulations.
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u


class Source(object):
    """
    """
    def __init__(self, wavelength, flux):
        self.wavelength = wavelength
        self.flux = flux


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
    return Source(config.wavelength, flux)
