# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an instrument response for spectroscopic simulations.
"""
from __future__ import print_function, division

import os.path
import glob
import math

import yaml

import numpy as np
import scipy.interpolate

import specsim.spectrum


class InstrumentX(object):
    """
    A class representing the DESI instrument for simulating observations
    """
    def __init__(self, modelFilename=None,throughputPath=None,psfFile=None,basePath=''):
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
            self.throughput[camera] = specsim.spectrum.WavelengthFunction.load(
                os.path.join(throughputPath,'thru-%s.fits' % camera),
                wavelength_column='wavelength', values_column='throughput',
                hdu=1, extrapolated_value=0.)
        # Loop over fiberloss models present in the throughput directory.
        self.fiberloss = { }
        for fiberlossFile in glob.iglob(os.path.join(throughputPath,'fiberloss-*.dat')):
            # Extract the model name from the filename.
            model = os.path.basename(fiberlossFile)[10:-4]
            # Load the data from this file.
            self.fiberloss[model] = specsim.spectrum.WavelengthFunction.load(
                fiberlossFile,extrapolated_value=0.)
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
            hdu = ('QUICKSIM-%s' % band.upper())
            # Load tabulated PSF functions of wavelength.
            self.angstromsPerRow[band] = specsim.spectrum.WavelengthFunction.load(
                psfFile, hdu=hdu, wavelength_column='wavelength',
                values_column='angstroms_per_row', extrapolated_value=0.)
            self.psfFWHMWavelength[band] = specsim.spectrum.WavelengthFunction.load(
                psfFile, hdu=hdu, wavelength_column='wavelength',
                values_column='fwhm_wave', extrapolated_value=0.)
            self.psfFWHMSpatial[band] = specsim.spectrum.WavelengthFunction.load(
                psfFile, hdu=hdu, wavelength_column='wavelength',
                values_column='fwhm_wave', extrapolated_value=0.)
            self.psfNPixelsSpatial[band] = specsim.spectrum.WavelengthFunction.load(
                psfFile, hdu=hdu, wavelength_column='wavelength',
                values_column='neff_spatial', extrapolated_value=0.)
            # Get the wavelength limits for the camera from the FITS header.
            wave = self.angstromsPerRow[band].wavelength
            waveMin, waveMax = wave[0], wave[-1]
            self.cameraWavelengthRanges.append((waveMin,waveMax))
            cameraMidpt.append(0.5*(waveMin+waveMax))
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


class Instrument(object):
    """
    """
    def __init__(self, name, fiber_acceptance, cameras, primary_mirror_diameter,
                 obscuration_diameter, support_width, fiber_diameter,
                 exposure_time):
        self.name = name
        self.fiber_acceptance = fiber_acceptance
        self.cameras = cameras
        self.primary_mirror_diameter = primary_mirror_diameter
        self.obscuration_diameter = obscuration_diameter
        self.support_width = support_width
        self.fiber_diameter = fiber_diameter
        self.exposure_time = exposure_time

        # Calculate the geometric area.
        D = self.primary_mirror_diameter
        obs = self.obscuration_diameter
        support_area = 0.5*(D - obs) * self.support_width
        self.effective_area = (
            math.pi * ((0.5 * D) ** 2 - (0.5 * obs) ** 2) - 4 * support_area)

        # Calculate the fiber area.
        self.fiber_area = math.pi * (0.5 * self.fiber_diameter) ** 2


class Camera(object):
    """
    """
    def __init__(self, name, wavelength, throughput, angstroms_per_row,
                 fwhm_wave, fwhm_spatial, neff_spatial, read_noise,
                 dark_current, gain):
        self.name = name
        self.wavelength = wavelength
        self.throughput = throughput
        self.angstroms_per_row = angstroms_per_row
        self.fwhm_wave = fwhm_wave
        self.fwhm_spatial = fwhm_spatial
        self.neff_spatial = neff_spatial
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.gain = gain


def initialize(config):
    """Initialize the instrument model from configuration parameters.

    Parameters
    ----------
    config : :class:`specsim.config.Configuration`
        The configuration parameters to use.

    Returns
    -------
    Instrument
        An initialized instrument model.
    """
    name = config.get('instrument.name').value
    cameras = config.get('instrument.cameras')
    camera_names = cameras.value.keys()
    initialized_cameras = []
    for camera_name in camera_names:
        camera = cameras.get(camera_name)
        psf = config.load_table(
            camera.get('psf'),
            ['angstroms_per_row', 'fwhm_wave', 'fwhm_spatial', 'neff_spatial'])
        throughput = config.load_table(
            camera.get('throughput'), 'throughput')
        constants = config.get_constants(camera,
            ['read_noise', 'dark_current', 'gain'])
        initialized_cameras.append(Camera(
            camera_name, config.wavelength, throughput,
            psf['angstroms_per_row'], psf['fwhm_wave'], psf['fwhm_spatial'],
            psf['neff_spatial'], constants['read_noise'],
            constants['dark_current'], constants['gain']))

    constants = config.get_constants(
        config.get('instrument'),
        ['exposure_time', 'primary_mirror_diameter', 'obscuration_diameter',
         'support_width', 'fiber_diameter'])

    fiber_acceptance = config.load_table(
        config.get('instrument.fiberloss'), 'fiber_acceptance')

    instrument = Instrument(
        name, fiber_acceptance, initialized_cameras,
        constants['primary_mirror_diameter'], constants['obscuration_diameter'],
        constants['support_width'], constants['fiber_diameter'],
        constants['exposure_time'])

    if config.verbose:
        print('Telescope effective area: {0}'.format(instrument.effective_area))
        print('Fiber entrance area: {0}'.format(instrument.fiber_area))

    return instrument
