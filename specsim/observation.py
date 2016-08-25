# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an astronomical observation for spectroscopic simulations.

An observation is usually initialized from a configuration used to create
a simulator and then accessible via its ``observation`` attribute, for example:

    >>> import specsim.simulator
    >>> simulator = specsim.simulator.Simulator('test')
    >>> print(simulator.observation.pointing)
    <SkyCoord (ICRS): (ra, dec) in deg
        (0.0, 0.0)>

After initialization, all aspects of an observation can be modified at runtime.
"""
from __future__ import print_function, division

import numpy as np

import astropy.coordinates
import astropy.units as u

import specsim.transform


class Observation(object):
    """Model the parameters describing a single spectroscopic observation.

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation
        Observatory location on the surface of the earth.
    exposure_time : astropy.units.Quantity
        Open shutter exposure time for this observation.
    pointing : astropy.coordinates.SkyCoord
        Sky position where the telescope boresight is pointing during the
        observation.
    wavelength : nastropy.units.Quantity
        Array of wavelength bin centers where the simulated spectrum is
        calculated, with units.
    timestamp : astropy.time.Time
        Time when the shutter opens and the exposure starts.
    """
    def __init__(self, location, exposure_time, exposure_start, pointing,
                 wavelength,
                 temperature=15*u.deg_C, pressure=None, relative_humidity=0):
        self.location = location
        self.exposure_time = exposure_time
        self.exposure_start = exposure_start
        self.pointing = pointing
        self.observing_model = specsim.transform.create_observing_model(
            self.location, self.exposure_start + 0.5 * self.exposure_time,
            wavelength, temperature, pressure, relative_humidity)


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
    constants = config.get_constants(config.observation, ['exposure_time'])
    location = specsim.transform.observatories[node.observatory]
    pointing = config.get_sky(node.pointing)
    exposure_start = config.get_timestamp(node.exposure_start)
    obs = Observation(
        location, constants['exposure_time'], exposure_start, pointing,
        config.wavelength)

    if config.verbose:
        print('Observatory located at {0}.'.format(obs.location))
        point = obs.pointing.transform_to('icrs')
        print('Observing field center (ra, dec) = ({0}, {1}).'.format(
            point.ra, point.dec))
        print('Exposure start MJD {0:.3f}, duration {1}.'.format(
            obs.exposure_start.mjd, obs.exposure_time))

    return obs
