# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model atmospheric emission and absorption for spectroscopic simulations.

The atmosphere model is responsible for calculating the spectral flux density
arriving at the telescope given a source flux entering the atmosphere. The
calculation is either performed as:

.. math::

    f(\lambda) = 10^{-e(\lambda) X / 2.5} s(\lambda) + a b(\lambda)

if ``extinct_emission`` is False, or else as:

.. math::

    f(\lambda) = 10^{-e(\lambda) X / 2.5} \left[
    s(\lambda) + a b(\lambda)\\right]

where :math:`s(\lambda)` is the source flux entering the atmosphere,
:math:`e(\lambda)` is the zenith extinction, :math:`X` is the airmass,
:math:`a` is the fiber entrance face area, and :math:`b(\lambda)` is the
sky emission surface brightness.
"""
from __future__ import print_function, division

import numpy as np


class Atmosphere(object):
    """Implement an atmosphere model based on tabulated data read from files.
    """
    def __init__(self, surface_brightness, extinction_coefficient,
                 extinct_emission, airmass):

        self.surface_brightness = surface_brightness
        self.extinction_coefficient = extinction_coefficient
        self.extinct_emission = extinct_emission
        self.set_airmass(airmass)


    def set_airmass(self, airmass):
        """
        """
        self.airmass = airmass
        self.extinction = 10 ** (-self.extinction_coefficient * airmass / 2.5)


    def propagate(self, source_flux, fiber_area):
        """Propagate a source flux through the atmosphere and into a fiber.
        """
        sky = self.surface_brightness * fiber_area
        if extinct_emission:
            sky *= self.extinction
        return sky + source_flux * self.extinction


def initialize(config):
    """Initialize the atmosphere model from configuration parameters.

    Parameters
    ----------
    config : :class:`specsim.config.Configuration`
        The configuration parameters to use.

    Returns
    -------
    Atmosphere
        An initialized atmosphere model.
    """
    # Check for required top-level config nodes.
    atmosphere = config.get('atmosphere')
    sky = atmosphere.get('sky')
    extinction = atmosphere.get('extinction')

    # Look up option values.
    extinct_emission = atmosphere.get('extinct_emission').value
    airmass = atmosphere.get('airmass').value

    # Load tabulated data.
    surface_brightness = config.load_table(sky, 'surface_brightness')
    extinction_coefficient = config.load_table(
        extinction, 'extinction_coefficient')

    return Atmosphere(
        surface_brightness, extinction_coefficient,
        extinct_emission, airmass)
