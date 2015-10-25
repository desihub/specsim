# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provides Atmosphere class for simulations of DESI observations
"""
from __future__ import print_function, division

import os.path
import numpy as np
import scipy.interpolate

import specsim.spectrum


class Atmosphere(object):
    def __init__(self,skySpectrumFilename=None,zenithExtinctionFilename=None,
        skyConditions='dark',basePath=''):
        """
        Initializes a new Atomosphere object using sky spectrum and zenith
        extinction values read from the specified text files. The sky spectrum
        file is required to have a linear wavelength scale. Comment lines beginning
        with # are ignored in the input files. If any filename is None, use a
        default relative to basePath, appropriate for skyConditions which should be
        one of 'dark','grey', or 'bright' (skyConditions is ignored when a skySpectrumFilename
        is provided).
        """
        # Apply defaults if necessary.
        if not skySpectrumFilename:
            skyNames = {
                'dark':'spec-sky.dat','grey':'spec-sky-grey.dat','bright':'spec-sky-bright.dat'}
            if skyConditions not in skyNames:
                raise RuntimeError('Atmosphere: invalid skyConditions "%s", expected one of %s.' %
                    (skyConditions,','.join(skyNames.keys())))
            skySpectrumFilename = os.path.join(basePath,
                'data','spectra',skyNames[skyConditions])
        if not zenithExtinctionFilename:
            zenithExtinctionFilename = os.path.join(basePath,
                'data','spectra','ZenithExtinction-KPNO.dat')
        # Load the tabulated sky spectrum.
        self.skySpectrum =\
            specsim.spectrum.SpectralFluxDensity.loadFromTextFile(skySpectrumFilename)
        # Load the tabulated zenith extinction coefficients.
        self.zenithExtinction =\
            specsim.spectrum.WavelengthFunction.loadFromTextFile(zenithExtinctionFilename)
