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


class Observation(object):
    """Model the parameters describing a single spectroscopic observation.

    Parameters
    ----------
    exposure_time : astropy.units.Quantity
        Open shutter exposure time for this observation.
    pointing : astropy.coordinates.SkyCoord
        Sky position where the telescope boresight is pointing during the
        observation.
    """
    def __init__(self, exposure_time, pointing):
        self.exposure_time = exposure_time
        self.pointing = pointing


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
    pointing = config.get_sky(node.pointing)
    return Observation(constants['exposure_time'], pointing)
