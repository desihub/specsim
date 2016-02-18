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

An atmosphere model is usually initialized from a configuration, for example:

    >>> import specsim.config
    >>> config = specsim.config.load_config('test')
    >>> atmosphere = initialize(config)
    >>> atmosphere.airmass
    1.0
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u


class Atmosphere(object):
    """Implement an atmosphere model based on tabulated data read from files.
    """
    def __init__(self, wavelength, surface_brightness_dict,
                 extinction_coefficient, extinct_emission, condition, airmass):
        self.wavelength = wavelength
        self.surface_brightness_dict = surface_brightness_dict
        self.extinction_coefficient = extinction_coefficient
        self.extinct_emission = extinct_emission
        self.condition_names = surface_brightness_dict.keys()

        self.set_condition(condition)
        self.set_airmass(airmass)


    def set_condition(self, name):
        """
        """
        if name not in self.condition_names:
            raise ValueError(
                "Invalid condition '{0}'. Pick one of {1}."
                .format(name, self.condition_names))
        self.condition = name
        self.surface_brightness = self.surface_brightness_dict[name]


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


    def plot(self):
        """Plot a summary of this atmosphere model.

        Requires that the matplotlib package is installed.
        """
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1_rhs = ax1.twinx()

        wave = self.wavelength.to(u.Angstrom).value
        wave_unit = u.Angstrom

        sky_unit = 1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom * u.arcsec**2)
        sky = self.surface_brightness.to(sky_unit).value
        sky_min, sky_max = np.percentile(sky, (1, 99))

        ext = self.extinction_coefficient
        ext_min, ext_max = np.percentile(ext, (1, 99))

        ax1.scatter(wave, sky, color='g', lw=0, s=1.)
        ax1_rhs.scatter(wave, ext, color='r', lw=0, s=1.)

        ax1.set_yscale('log')
        ax1_rhs.set_yscale('log')

        ax1.set_ylabel(
            'Surface Brightness [$10^{-17}\mathrm{erg}/(\mathrm{cm}^2' +
            '\mathrm{s} \AA)/\mathrm{arcsec}^2$]')
        ax1.set_ylim(0.5 * sky_min, 1.5 * sky_max)
        ax1_rhs.set_ylabel('Zenith Extinction')
        ax1_rhs.set_ylim(0.5 * ext_min, 1.5 * ext_max)

        ax1.set_xlabel('Wavelength [$\AA$]')
        ax1.set_xlim(wave[0], wave[-1])

        ax1.plot([], [], 'g-',
                 label='Surface Brightness ({0})'.format(self.condition))
        ax1.plot([], [], 'r-', label='Zenith Extinction Coefficient')
        ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)


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
    atm_config = config.atmosphere

    # Load tabulated data.
    surface_brightness_dict = config.load_table(
        atm_config.sky, 'surface_brightness', as_dict=True)
    extinction_coefficient = config.load_table(
        atm_config.extinction, 'extinction_coefficient')

    atmosphere = Atmosphere(
        config.wavelength, surface_brightness_dict, extinction_coefficient,
        atm_config.extinct_emission, atm_config.sky.condition,
        atm_config.airmass)

    if config.verbose:
        print(
            "Atmosphere initialized with condition '{0}' from {1}."
            .format(atmosphere.condition, atmosphere.condition_names))

    return atmosphere
