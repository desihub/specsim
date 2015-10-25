# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provides functions for creating different source spectra.
"""
from __future__ import print_function, division

import math

import numpy as np

root2pi = np.sqrt(2*math.pi)


def getOIIDoubletFlux(totalFlux,wave):
    """
    Returns a vector of [OII] doublet fluxes in erg/(s*cm^2*Ang) corresponding to the
    input vector of rest-frame wavelengths in Angstroms. Uses the model of:
    https://desi.lbl.gov/trac/wiki/Pipeline/QuickSim#ELGs
    with a line ratio of 1:1.3, and an intrinsic velocity dispersion of 70 km/s.
    """
    lam1,lam2 = 3727.0917,3729.8754
    sigma = 70./3e5*lam1
    arg1 = (wave - lam1)/sigma
    arg2 = (wave - lam2)/sigma
    return totalFlux/(sigma*root2pi)*(
        (1.0/2.3)*np.exp(-0.5*arg1**2) + (1.3/2.3)*np.exp(-0.5*arg2**2))

lyaSpectra = None
lyaZ = None

def loadLyaSpectra(filename,zvec):
    """
    Loads the Lya spectra tabulated at different redshifts from the specified file.
    """
    # Load the tabulated wavelengths and fluxes from the file.
    data = np.loadtxt(lyaFilename,unpack=True)
    wave,fluxes = data[0],data[1:]
    assert len(zvec) == len(fluxes),(
        "loadLyaSpectra: redshift parameters do match file shape: %d,%d" %
        (len(zvec),len(fluxes)))
    # Save the redshift vector.
    lyaZ = numpy.array(zvec)
    # Loop over redshifts and save a corresponding spectrum.
    lyaSpectra = [ ]
    for z in zvec:
        lyaSpectra.append(sim.SpectralFluxDensity(wave,fluxes[i],extrapolatedValue=0.))

names = ['[OII]','LyaQSO']

def createSource(name,redshift,totalFlux=None,magnitude=None,band='g'):
    """
    Returns a SpectralFluxDensity object for the named source type at the
    specified redshift. The flux normalization must be specified using either
    the totalFlux in erg/(s*cm^2*Ang) or AB magnitude parameters, but not both.
    The default magnitude is SDSS g band, but other SDSS bands can be specified
    as long as they are covered by the underlying source model. The module
    global sources.names lists the recognized source names.
    """
    if name not in names:
        raise RuntimeError('createSource: name %r not supported' % name)
