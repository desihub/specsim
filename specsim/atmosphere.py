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
    def __init__(self, skySpectrumFilename=None, zenithExtinctionFilename=None,
                 skyConditions='dark', basePath=''):
        # Apply defaults if necessary.
        if not skySpectrumFilename:
            skyNames = {
                'dark' : 'spec-sky.dat',
                'grey' : 'spec-sky-grey.dat',
                'bright' : 'spec-sky-bright.dat'}
            if skyConditions not in skyNames:
                raise ValueError('Atmosphere: invalid skyConditions "{0}",' +
                    ' expected one of {1}.'
                    .format(skyConditions, ','.join(skyNames.keys())))
            skySpectrumFilename = os.path.join(
                basePath, 'data', 'spectra', skyNames[skyConditions])
        if not zenithExtinctionFilename:
            zenithExtinctionFilename = os.path.join(
                basePath, 'data', 'spectra', 'ZenithExtinction-KPNO.dat')
        # Load the tabulated sky spectrum.
        self.skySpectrum =\
            specsim.spectrum.SpectralFluxDensity.loadFromTextFile(skySpectrumFilename)
        # Load the tabulated zenith extinction coefficients.
        self.zenithExtinction =\
            specsim.spectrum.WavelengthFunction.loadFromTextFile(zenithExtinctionFilename)
