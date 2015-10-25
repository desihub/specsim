# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provides Instrument class for simulations of DESI observations
"""
from __future__ import print_function, division


import os.path
import glob
import yaml
import numpy as np
from astropy.io import fits
import scipy.interpolate

import specsim.spectrum


class Instrument(object):
    """
    A class representing the DESI instrument for simulating observations
    """
    def __init__(self,modelFilename=None,throughputPath=None,psfFile=None,basePath=''):
        """
        Initializes an Instrument using parameters in the specified yaml model file,
        throughputs in the specified path, and psf parameters in the specified
        FITS file. If any filename or path is None, use a default relative to basePath.
        """
        # Apply defaults if necessary.
        if modelFilename is None:
            modelFilename = os.path.join(basePath,'data','desi.yaml')
        if throughputPath is None:
            throughputPath = os.path.join(basePath,'data','throughput')
        if psfFile is None:
            psfFile = os.path.join(basePath,'data','specpsf','psf-quicksim.fits')
        # Load the model parameters from the specified YAML file.
        if not os.path.isfile(modelFilename):
            raise RuntimeError('No such yaml model file: %s' % modelFilename)
        with open(modelFilename) as stream:
            self.model = yaml.safe_load(stream)
        # Load the throughputs for each camera from the specified path.
        if not os.path.isdir(throughputPath):
            raise RuntimeError('No such throughput path: %s' % throughputPath)
        # Loop over cameras listed in the model.
        self.throughput = { }
        for camera in self.model['ccd']:
            # Try to open the throughput FITS file for this camera.
            hduList = fits.open(os.path.join(throughputPath,'thru-%s.fits' % camera))
            # Create linear interpolations of the wavelength and throughput columns
            # from the FITS table stored as HDU[1]. Values outside of the tabulated
            # wavelength range will silently extrapolate to zero.
            table = hduList[1].data
            self.throughput[camera] = specsim.spectrum.WavelengthFunction(
                table['wavelength'],table['throughput'], extrapolatedValue=0.)
            hduList.close()
        # Loop over fiberloss models present in the throughput directory.
        self.fiberloss = { }
        for fiberlossFile in glob.iglob(os.path.join(throughputPath,'fiberloss-*.dat')):
            # Extract the model name from the filename.
            model = os.path.basename(fiberlossFile)[10:-4]
            # Load the data from this file.
            self.fiberloss[model] = specsim.spectrum.WavelengthFunction.loadFromTextFile(
                fiberlossFile,extrapolatedValue=0.)
        # Open the PSF parameter file.
        hduList = fits.open(psfFile)
        # Loop over camera bands to build linear interpolations of the PSF FWHM in the
        # wavelength (Angstroms) and spatial (pixels) directions. We also build an
        # interpolation of the angstromsPerRow values needed to convert between pixels
        # and Angstroms in the wavelength direction.
        self.angstromsPerRow = { }
        self.psfFWHMWavelength = { }
        self.psfFWHMSpatial = { }
        self.psfNPixelsSpatial = { }
        self.cameraBands = self.model['ccd'].keys()
        self.cameraWavelengthRanges = [ ]
        cameraMidpt = [ ]
        for band in self.cameraBands:
            # Use a key of the form QUICKSIM-X where X identifies the camera band.
            # Note that FITS does not recognize unicode strings as keys so we must
            # explicity encode in 'ascii' here (or not import unicode_literals above).
            key = ('QUICKSIM-%s' % band.upper()).encode('ascii')
            table = hduList[key].data
            wave = table['wavelength']
            # Load tabulated PSF functions of wavelength.
            self.angstromsPerRow[band] = specsim.spectrum.WavelengthFunction(
                wave,table['angstroms_per_row'],extrapolatedValue=0.)
            self.psfFWHMWavelength[band] = specsim.spectrum.WavelengthFunction(
                wave,table['fwhm_wave'],extrapolatedValue=0.)
            self.psfFWHMSpatial[band] = specsim.spectrum.WavelengthFunction(
                wave,table['fwhm_wave'],extrapolatedValue=0.)
            self.psfNPixelsSpatial[band] = specsim.spectrum.WavelengthFunction(
                wave,table['neff_spatial'],extrapolatedValue=0.)
            # Get the wavelength limits for the camera from the FITS header.
            waveMin,waveMax = hduList[key].header['WMIN_ALL'],hduList[key].header['WMAX_ALL']
            assert waveMin == wave[0] and waveMax == wave[-1], (
                "Inconsistent wavelength limits for %s" % key)
            self.cameraWavelengthRanges.append((waveMin,waveMax))
            cameraMidpt.append(0.5*(waveMin+waveMax))
        hduList.close()
        # Sort the camera bands in order of increasing wavelength (blue to red). This info
        # is already encoded in the order in which the cameras appear in the YAML file, but
        # is not preserved a YAML parser, which views the 'ccd' record as a dictionary.
        self.cameraBands = [x for (y,x) in sorted(zip(cameraMidpt,self.cameraBands))]
        self.cameraWavelengthRanges = [x for (y,x) in sorted(zip(cameraMidpt,self.cameraWavelengthRanges))]

    def getSourceTypes(self):
        """
        Returns a list of the source types for which we have a fiber loss model defined.
        """
        return self.fiberloss.keys()

    def getCameraBands(self):
        """
        Returns a list of the camera bands supported by this instrument, in order of
        increasing wavelength.
        """
        return self.cameraBands
