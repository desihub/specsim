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
                 extinction_coefficient, extinct_emission, condition, airmass,
                 moon):
        self.wavelength = wavelength
        self.surface_brightness_dict = surface_brightness_dict
        self.extinction_coefficient = extinction_coefficient
        self.extinct_emission = extinct_emission
        self.condition_names = surface_brightness_dict.keys()
        self.moon = moon

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


class Moon(object):
    """Model of scattered moonlight.
    """
    def __init__(self, moon_spectrum, extinction_coefficient, moon_phase,
                 moon_zenith, observation_zenith, separation_angle):
        self.moon_spectrum = moon_spectrum
        self.extinction_coefficient = extinction_coefficient
        self.update(
            moon_phase, moon_zenith, observation_zenith, separation_angle)


    def update(self, moon_phase, moon_zenith, observation_zenith,
               separation_angle):
        self.moon_phase = moon_phase
        self.moon_zenith = moon_zenith
        self.observation_zenith = observation_zenith
        self.separation_angle = separation_angle


def krisciunas_schaefer(obs_zenith, moon_zenith, separation_angle, moon_phase,
                        extinction_coefficient):
    """Calculate the moon surface brightness.

    Based on Krisciunas and Schaefer, "A model of the brightness of moonlight",
    PASP, vol. 103, Sept. 1991, p. 1033-1039 (http://dx.doi.org/10.1086/132921).
    Equation numbers in the code comments refer to this paper.

    This method has many caveats but the authors find agreement with data at
    the 8% - 23% level.  See the paper for details.

    Parameters
    ----------
    obs_zenith : astropy.units.Quantity
        Zenith angle of the observation in angular units.
    moon_zenith : astropy.units.Quantity
        Zenith angle of the moon in angular units.
    separation_angle : astropy.units.Quantity
        Opening angle between the observation and moon in angular units.
    moon_phase : float
        Phase of the moon from 0.0 (full) to 1.0 (new), which can be calculated
        as abs((d / D) - 1) where d is the time since the last new moon
        and D = 29.5 days is the period between new moons.
    extinction_coefficient : float
        V-band extinction coefficient to use.

    Returns
    -------
    astropy.units.Quantity
        Observed V-band surface brightness of scattered moonlight.
    """
    moon_phase = np.asarray(moon_phase)
    if np.any((moon_phase < 0) | (moon_phase > 1)):
        raise ValueError(
            'Invalid moon phase {0}. Expected 0-1.'.format(moon_phase))
    # Calculate the V-band magnitude of the moon (eqn. 9).
    abs_alpha = 180. * moon_phase
    m = -12.73 + 0.026 * abs_alpha + 4e-9 * abs_alpha ** 4
    # Calculate the illuminance of the moon outside the atmosphere in
    # foot-candles (eqn. 8).
    Istar = 10 ** (-0.4 * (m + 16.57))
    # Calculate the scattering function (eqn.21).
    rho = separation_angle.to(u.deg).value
    f_scatter = (10 ** 5.36 * (1.06 + np.cos(separation_angle) ** 2) +
                 10 ** (6.15 - rho / 40.))
    # Calculate the scattering airmass along the lines of sight to the
    # observation and moon (eqn. 3).
    X_obs = np.sqrt(1 - 0.96 * np.sin(obs_zenith) ** 2)
    X_moon = np.sqrt(1 - 0.96 * np.sin(moon_zenith) ** 2)
    # Calculate the V-band moon surface brightness in nanoLamberts.
    B_moon = (f_scatter * Istar *
        10 ** (-0.4 * extinction_coefficient * X_moon) *
        (1 - 10 ** (-0.4 * (extinction_coefficient * X_obs))))
    # Convert from nanoLamberts to to mag / arcsec**2 using eqn.19 of
    # Garstang, "Model for Artificial Night-Sky Illumination",
    # PASP, vol. 98, Mar. 1986, p. 364 (http://dx.doi.org/10.1086/131768)
    return ((20.7233 - np.log(B_moon / 34.08)) / 0.92104 *
            u.mag / (u.arcsec ** 2))


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

    # Initialize an optional lunar scattering model.
    moon_config = getattr(atm_config, 'moon', None)
    if moon_config:
        moon_spectrum = config.load_table(moon_config, 'flux')
        c = config.get_constants(moon_config,
            ['moon_phase', 'moon_zenith', 'observation_zenith',
             'separation_angle'])
        moon = Moon(
            moon_spectrum, extinction_coefficient, c['moon_phase'],
            c['moon_zenith'], c['observation_zenith'], c['separation_angle'])
    else:
        moon = None

    atmosphere = Atmosphere(
        config.wavelength, surface_brightness_dict, extinction_coefficient,
        atm_config.extinct_emission, atm_config.sky.condition,
        atm_config.airmass, moon)

    if config.verbose:
        print(
            "Atmosphere initialized with condition '{0}' from {1}."
            .format(atmosphere.condition, atmosphere.condition_names))

    return atmosphere
