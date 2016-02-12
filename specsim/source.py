# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an astronomical source for spectroscopic simulations.
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u


class Source(object):
    """
    """
    def __init__(self):
        pass


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
    source = config.get('source')

    return Source()
