# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an astronomical observation for spectroscopic simulations.

An observation is usually initialized from a configuration used to create
a simulator and then accessible via its ``observation`` attribute, for example:

    >>> import specsim.simulator
    >>> simulator = specsim.simulator.Simulator('test')  # doctest: +IGNORE_OUTPUT
    >>> print(simulator.observation.exposure_time)
    1000.0 s

"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u
import astropy.coordinates

import specsim.transform


class Observation(object):
    """Model the parameters describing a single spectroscopic observation.

    The following parameters can be changed after an Observation object has
    been created: exposure_time, exposure_start, pointing.

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation
        Observatory location on the surface of the earth.
    exposure_time : astropy.units.Quantity
        Open shutter exposure time for this observation.
    exposure_start : astropy.time.Time
        Time when the shutter opens and the exposure starts.
    pointing : astropy.coordinates.SkyCoord
        Sky position where the telescope boresight is pointing during the
        observation.
    wavelength : astropy.units.Quantity
        Array of wavelength bin centers where the simulated spectrum is
        calculated, with units.
    pressure : astropy.units.Quantity
        Used to create an :func:`observing model
        <specsim.transform.create_observing_model>`.
    temperature : astropy.units.Quantity
        Used to create an :func:`observing model
        <specsim.transform.create_observing_model>`.
    relative_humidity : astropy.units.Quantity
        Used to create an :func:`observing model
        <specsim.transform.create_observing_model>`.
    """
    def __init__(self, location, exposure_time, exposure_start, pointing,
                 wavelength, pressure, temperature, relative_humidity):
        self.central_wavelength = 0.5 * (wavelength[0] + wavelength[-1])
        self._exposure_time = exposure_time
        self._location = location
        self._exposure_start = exposure_start
        self._pointing = pointing
        self._temperature = temperature
        self._pressure = pressure
        self._relative_humidity = relative_humidity
        self._update_model()


    def _update_model(self):
        """Update observing model. Triggered internally.
        """
        # Initialize an observing model at the middle of the exposure and
        # at the central wavelength of the simulation, i.e., ignore temporal
        # and chromatic variations (for now).
        exposure_midpoint = self.exposure_start + 0.5 * self.exposure_time
        self.observing_model = specsim.transform.create_observing_model(
            self.location, exposure_midpoint, self.central_wavelength,
            self.temperature, self.pressure, self.relative_humidity)
        # Calculate the boresight angles (fixed, since we do not consider
        # temporal or chromatic effects yet).
        self.boresight_altaz = specsim.transform.sky_to_altaz(
            self.pointing, self.observing_model)


    @property
    def location(self):
        """astropy.coordinates.EarthLocation: Observatory location.
        """
        return self._location


    @property
    def exposure_time(self):
        """astropy.units.Quantity: Exposure time for this observation.
        """
        return self._exposure_time


    @exposure_time.setter
    def exposure_time(self, value):
        try:
            self._exposure_time = value.to(u.s)
            self._update_model()
        except (AttributeError, u.UnitConversionError):
            raise ValueError('Invalid units for exposure_time.')


    @property
    def exposure_start(self):
        """astropy.time.Time: Time when the shutter opens.
        """
        return self._exposure_start


    @exposure_start.setter
    def exposure_start(self, value):
        self._exposure_start = value
        self._update_model()


    @property
    def pointing(self):
        """astropy.coordinates.SkyCoord: Telescope boresight pointing.
        """
        return self._pointing


    @pointing.setter
    def pointing(self, value):
        self._pointing = value
        self._update_model()


    @property
    def temperature(self):
        """astropy.units.Quantity: Air temperature at observatory.

        Used for atmospheric refraction model.
        """
        return self._temperature


    @property
    def pressure(self):
        """astropy.units.Quantity: Atmospheric pressure at observatory.

        Used for atmospheric refraction model.
        """
        return self._pressure


    @property
    def relative_humidity(self):
        """float: Relative humidity at observatory.

        Used for atmospheric refraction model.
        """
        return self._relative_humidity


    def locate_on_focal_plane(self, sky_position, instrument):
        """Transform a sky position into focal-plane coordinates.

        The input sky position can refer to one or more objects.

        Parameters
        ----------
        sky_position : astropy.coordinates.SkyCoord
            Position of one or more objects on the sky.
        instrument : specsim.instrument.Instrument
            Description of the instrument being simulated.

        Returns
        -------
        tuple
            Tuple (x, y) of focal-plane coordinates given as astropy
            quantities with length units.
        """
        altaz = specsim.transform.sky_to_altaz(
            sky_position, self.observing_model)
        # Calculate field angles relative to the boresight.
        x, y = specsim.transform.altaz_to_focalplane(
            altaz.alt, altaz.az,
            self.boresight_altaz.alt, self.boresight_altaz.az)
        # Convert field angles to focal-plane coordinates.
        angle = np.sqrt(x ** 2 + y ** 2)
        scale = np.zeros(angle.shape) * u.mm / u.deg
        nonzero = angle > 0
        try:
            scale[nonzero] = (
                instrument.field_angle_to_radius(angle[nonzero]) /
                angle[nonzero])
        except TypeError:
            if nonzero:
                scale = instrument.field_angle_to_radius(angle) / angle
        x = (x * scale).to(u.mm)
        y = (y * scale).to(u.mm)
        return x, y


def initialize(config):
    """Initialize the observation from configuration parameters.

    Parameters
    ----------
    config : :class:`specsim.config.Configuration`
        The configuration parameters to use.

    Returns
    -------
    Observation
        An initialized observation.
    """
    node = config.observation
    constants = config.get_constants(
        config.observation,
        ['exposure_time', 'temperature', 'relative_humidity'],
        optional_names=['pressure'])
    pressure = constants.get('pressure', None)
    location = specsim.transform.observatories[node.observatory]
    pointing = config.get_sky(node.pointing)
    exposure_start = config.get_timestamp(node.exposure_start)
    adjust_ha = getattr(node.exposure_start, 'adjust_to_hour_angle', None)
    if adjust_ha is not None:
        nominal_start = exposure_start
        point_radec = pointing.transform_to('icrs')
        hour_angle = astropy.coordinates.Angle(adjust_ha)
        try:
            lon = location.lon
        except AttributeError:
            # Required for astropy < 2.0
            lon = location.longitude
        exposure_start = specsim.transform.adjust_time_to_hour_angle(
            nominal_start, point_radec.ra, hour_angle, lon)
        # Put the requested HA at the middle of the exposure.
        exposure_start -= 0.5 * constants['exposure_time']

    obs = Observation(
        location, constants['exposure_time'], exposure_start, pointing,
        config.wavelength, pressure, constants['temperature'],
        constants['relative_humidity'])

    if config.verbose:
        print('Observatory located at (lat, lon, elev) = ',
              '({0:.1f}, {1:.1f}, {2:.1f}).'
              .format(*obs.location.to_geodetic()))
        point = obs.pointing.transform_to('icrs')
        print('Observing field center (ra, dec) = ({0}, {1}).'.format(
            point.ra, point.dec))
        print('Exposure start MJD {0:.3f}, duration {1}.'.format(
            obs.exposure_start.mjd, obs.exposure_time))
        if adjust_ha is not None:
            dt = exposure_start - nominal_start
            print('Adjusted by {0:+.3f} for HA {1}.'
                  .format(dt.to(u.hour), hour_angle))
        cond = obs.observing_model
        print('Conditions: pressure {0:.1f}, temperature {1:.1f}, RH {2}.'
              .format(cond.pressure, cond.temperature, cond.relative_humidity))
        altaz = obs.boresight_altaz
        print('Boresight (alt, az) = ({0:.1f}, {1:.1f}).'
              .format(altaz.alt, altaz.az))

    return obs
