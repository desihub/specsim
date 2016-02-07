# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model atmospheric emission and absorption for spectroscopic simulations.
"""
from __future__ import print_function, division

import os.path
import numpy as np
import scipy.interpolate

import specsim.spectrum


class Atmosphere(object):
    """Implement an atmosphere model based on tabulated data read from files.

    Files are read using :meth:`astropy.table.Table.read` so should have
    compatible extensions and formats.  The only exception is that the
    '.dat' extension will be interpreted with `format='ascii'` (since astropy
    does not do this automatically).

    Parameters
    ----------
    skySpectrumFilename : str or None
        Read the sky emission spectrum from the specified filename or else
        use a default spectrum if this parameter is None.
    zenithExtinctionFilename : str or None
        Read the atomspheric extinction function from the specified filename or
        else use a default function if this parameter is None.
    skyConditions : str
        Must be one of 'dark', 'grey', or 'bright'. Specifies the default
        sky emission spectrum file to use when the skySpectrumFilename parameter
        is None, or otherwise ignored.
    basePath : str
        Base path containing the default files used when skySpectrumFilename
        or zenithExtinctionFilename is None.

    Raises
    ------
    ValueError
        Invalid skyConditions.

    Attributes
    ----------
    skySpectrum : :class:`specsim.spectrum.SpectralFluxDensity`
        Tabulated sky emission spectral flux density per unit wavelength.
    zenithExtinction : :class:`specsim.spectrum.WavelengthFunction`
        Tabulated atmospheric extinction function at zenith (airmass = 1).
    """
    def __init__(self, sky_spectrum, zenith_extinction,
                 extinct_emission, airmass):

        self.skySpectrum = sky_spectrum
        self.zenithExtinction = zenith_extinction

        self.extinct_emission = extinct_emission
        self.airmass = airmass


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
    extinct_emission = atmosphere.get('extinct_emission').value
    airmass = atmosphere.get('airmass').value

    # Load tabulated data.
    sky_spectrum = config.load_table(sky.get('table'))
    zenith_extinction = config.load_table(extinction.get('table'))

    return Atmosphere(
        sky_spectrum, zenith_extinction, extinct_emission, airmass)
