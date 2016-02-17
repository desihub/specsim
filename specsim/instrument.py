# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an instrument response for spectroscopic simulations.

An instrument model is usually initialized from a configuration, for example:

    >>> import specsim.config
    >>> config = specsim.config.load_config('test')
    >>> instrument = initialize(config)
    >>> print(np.round(instrument.exposure_time, 1))
    1000.0 s
"""
from __future__ import print_function, division

import math

import numpy as np

import astropy.constants
import astropy.units as u


class Instrument(object):
    """
    """
    def __init__(self, name, wavelength, fiber_acceptance_dict, cameras,
                 primary_mirror_diameter, obscuration_diameter, support_width,
                 fiber_diameter, exposure_time):
        self.name = name
        self.wavelength = wavelength
        self.fiber_acceptance_dict = fiber_acceptance_dict
        self.cameras = cameras
        self.primary_mirror_diameter = primary_mirror_diameter
        self.obscuration_diameter = obscuration_diameter
        self.support_width = support_width
        self.fiber_diameter = fiber_diameter
        self.exposure_time = exposure_time

        self.source_types = self.fiber_acceptance_dict.keys()

        # Calculate the geometric area.
        D = self.primary_mirror_diameter
        obs = self.obscuration_diameter
        support_area = 0.5*(D - obs) * self.support_width
        self.effective_area = (
            math.pi * ((0.5 * D) ** 2 - (0.5 * obs) ** 2) - 4 * support_area)

        # Calculate the fiber area.
        self.fiber_area = math.pi * (0.5 * self.fiber_diameter) ** 2

        # Calculate the energy per photon at each wavelength.
        hc = astropy.constants.h * astropy.constants.c
        energy_per_photon = (hc / self.wavelength).to(u.erg)

        # Calculate the rate of photons incident on the focal plane per
        # wavelength bin per unit spectral flux density. The fiber acceptance
        # fraction is not included in this calculation.
        wavelength_bin_size = np.gradient(self.wavelength)
        self.photons_per_bin = (
            self.effective_area * wavelength_bin_size / energy_per_photon
            ).to((u.cm**2 * u.Angstrom) / u.erg)

        wave_mid = []
        for i, camera in enumerate(self.cameras):
            wave_min, wave_max = camera.wavelength_min, camera.wavelength_max
            wave_mid.append(0.5 * (wave_min + wave_max))
            if i == 0:
                self.wavelength_min = wave_min
                self.wavelength_max = wave_max
            else:
                self.wavelength_min = min(self.wavelength_min, wave_min)
                self.wavelength_max = max(self.wavelength_max, wave_max)

        # Sort cameras in order of increasing wavelength.
        self.cameras = [x for (y, x) in sorted(zip(wave_mid, self.cameras))]


    def get_fiber_acceptance(self, source):
        """
        """
        if source.type_name not in self.source_types:
            raise ValueError(
                "Invalid source type '{0}'. Pick one of {1}."
                .format(source.type_name, self.source_types))
        return self.fiber_acceptance_dict[source.type_name]


    def plot(self, flux=1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom),
             exposure_time=None, cmap='nipy_spectral'):
        """Plot a summary of this instrument's model.

        Requires that the matplotlib package is installed.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 8))
        ax1_rhs = ax1.twinx()
        ax2_rhs = ax2.twinx()
        cmap = cm.get_cmap(cmap)

        wave = self.wavelength.value
        wave_unit = self.wavelength.unit
        dwave = np.gradient(self.wavelength).value

        if exposure_time is None:
            exposure_time = self.exposure_time

        ax1.plot(wave, self.fiber_acceptance, 'k--')
        for camera in self.cameras:
            cwave = camera.wavelength.to(wave_unit).value

            # Use an approximate spectral color for each band.
            mid_wave = 0.5 * (camera.wavelength_min + camera.wavelength_max)
            color = cmap(
                (mid_wave - self.wavelength_min) /
                (self.wavelength_max - self.wavelength_min))

            nphot = (flux * self.photons_per_bin * exposure_time *
                     self.fiber_acceptance * camera.throughput / dwave)
            dark_noise = np.sqrt(
                (camera.dark_current_per_bin * exposure_time).value)
            total_noise = np.sqrt(
                dark_noise ** 2 + camera.read_noise_per_bin.value ** 2)

            ax1.plot(cwave, camera.throughput, ls='-', color=color)

            ax1_rhs.plot(cwave, nphot.value, ls=':', color=color)
            ax1_rhs.fill_between(
                cwave, total_noise / dwave, lw=0, color=color, alpha=0.2)
            ax1_rhs.fill_between(
                cwave, dark_noise / dwave, lw=0, color=color, alpha=0.2)
            ax1_rhs.plot(cwave, total_noise / dwave, ls='-.', color=color)

            ax2.plot(
                cwave, camera.rms_resolution.to(wave_unit).value,
                ls='-', color=color)
            ax2.plot(
                cwave, camera.row_size.to(wave_unit / u.pixel).value,
                ls='--', color=color)

            ax2_rhs.plot(cwave, camera.neff_spatial, ls=':', color=color)

        ax1.plot([], [], 'k--', label='Fiber Acceptance')
        ax1.plot([], [], 'k-', label='Camera Throughput')
        ax1.plot([], [], 'k:', label='{0}'.format(flux))
        ax1.plot([], [], 'k-.', label='Total Noise')
        ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        ax2.plot([], [], 'k-', label='RMS Resolution')
        ax2.plot([], [], 'k--', label='Row Size')
        ax2.plot([], [], 'k:', label='Column Size')
        ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)

        ax1.set_ylim(0., None)
        ax1.set_ylabel('Fiber, Camera Throughput')
        ax1_rhs.set_ylim(0., None)
        ax1_rhs.set_ylabel(
            'Photons, Electrons / Exposure / {0}'.format(wave_unit))
        ax2.set_ylim(0., None)
        ax2.set_ylabel('RMS Resolution, Row Size [{0}]'.format(wave_unit))
        ax2_rhs.set_ylim(0., None)
        ax2_rhs.set_ylabel('Effective Column Size [pixels]')
        ax2.set_xlabel('Wavelength [{0}]'.format(wave_unit))
        ax2.set_xlim(wave[0], wave[-1])


class Camera(object):
    """
    """
    def __init__(self, name, wavelength, throughput, row_size,
                 fwhm_resolution, neff_spatial, read_noise, dark_current, gain):
        self.name = name
        self.wavelength = wavelength
        self.throughput = throughput
        self.row_size = row_size
        self.fwhm_resolution = fwhm_resolution
        self.neff_spatial = neff_spatial
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.gain = gain

        # Calculate the RMS resolution assuming a Gaussian PSF.
        fwhm_to_sigma = 1. / (2 * math.sqrt(2 * math.log(2)))
        self.rms_resolution = fwhm_to_sigma * self.fwhm_resolution

        # The CCD properties should all have consistent coverage.
        ccd_nonzero = np.where(self.row_size > 0)[0]
        ccd_first, ccd_last = ccd_nonzero[0], ccd_nonzero[-1] + 1
        if (np.any(self.fwhm_resolution[:ccd_first] != 0) or
            np.any(self.fwhm_resolution[ccd_last:] != 0)):
            raise RuntimeError('Resolution extends beyond CCD coverage.')
        if (np.any(self.neff_spatial[:ccd_first] != 0) or
            np.any(self.neff_spatial[ccd_last:] != 0)):
            raise RuntimeError('Spatial Neff extends beyond CCD coverage.')
        if np.any(self.row_size[ccd_first:ccd_last] <= 0.):
            raise RuntimeError('CCD coverage has holes.')
        if np.any(self.fwhm_resolution[ccd_first:ccd_last] <= 0.):
            raise RuntimeError('CCD resolution has holes.')
        if np.any(self.neff_spatial[ccd_first:ccd_last] <= 0.):
            raise RuntimeError('CCD spatial Neff has holes.')

        #self.ccd_slice = slice(ccd_first, ccd_last)
        self.ccd_coverage = np.zeros_like(self.wavelength.value, dtype=bool)
        self.ccd_coverage[ccd_first:ccd_last] = True
        self.wavelength_min = self.wavelength[ccd_first]
        self.wavelength_max = self.wavelength[ccd_last - 1]

        # The camera throughput should have no holes and extend beyond the
        # CCD coverage to allow for dispersion at the edges.
        thru_nonzero = np.where(self.throughput > 0)[0]
        thru_first, thru_last = thru_nonzero[0], thru_nonzero[-1] + 1
        if np.any(self.throughput[thru_first:thru_last] <= 0.):
            raise RuntimeError('Camera throughput has holes.')
        thru_min = self.wavelength[thru_first]
        thru_max = self.wavelength[thru_last - 1]
        if thru_min > self.wavelength_min - self.rms_resolution[ccd_first]:
            raise RuntimeError('Throughput does not allow for edge dispersion.')
        if thru_max < self.wavelength_max + self.rms_resolution[ccd_last - 1]:
            raise RuntimeError('Throughput does not allow for edge dispersion.')

        # Calculate the size of each wavelength bin in units of pixel rows.
        wavelength_bin_size = np.gradient(self.wavelength)
        mask = self.row_size.value > 0
        neff_wavelength = np.zeros_like(self.neff_spatial)
        neff_wavelength[mask] = (
            wavelength_bin_size[mask] / self.row_size[mask]
            ).to(u.pixel)

        # Calculate the effective pixel area contributing to the signal
        # in each wavelength bin.
        self.neff_pixels = (
            neff_wavelength * self.neff_spatial).to(u.pixel ** 2)

        # Calculate the read noise per wavelength bin, assuming that
        # readnoise is uncorrelated between pixels (hence the sqrt scaling). The
        # value will be zero in pixels that are not used by this camera.
        self.read_noise_per_bin = (
            self.read_noise * np.sqrt(self.neff_pixels.value)
            ).to(u.electron)

        # Calculate the dark current per wavelength bin.
        self.dark_current_per_bin = (
            self.dark_current * self.neff_pixels).to(u.electron / u.s)


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
    name = config.instrument.name
    cameras = config.instrument.cameras
    camera_names = cameras.keys()
    initialized_cameras = []
    for camera_name in camera_names:
        camera = getattr(cameras, camera_name)
        ccd = config.load_table(
            camera.ccd, ['row_size', 'fwhm_resolution', 'neff_spatial'])
        throughput = config.load_table(camera.throughput, 'throughput')
        constants = config.get_constants(camera,
            ['read_noise', 'dark_current', 'gain'])
        initialized_cameras.append(Camera(
            camera_name, config.wavelength, throughput,
            ccd['row_size'], ccd['fwhm_resolution'],
            ccd['neff_spatial'], constants['read_noise'],
            constants['dark_current'], constants['gain']))

    constants = config.get_constants(
        config.instrument,
        ['exposure_time', 'primary_mirror_diameter', 'obscuration_diameter',
         'support_width', 'fiber_diameter'])

    fiber_acceptance_dict = config.load_table(
        config.instrument.fiberloss, 'fiber_acceptance', as_dict=True)

    instrument = Instrument(
        name, config.wavelength, fiber_acceptance_dict, initialized_cameras,
        constants['primary_mirror_diameter'], constants['obscuration_diameter'],
        constants['support_width'], constants['fiber_diameter'],
        constants['exposure_time'])

    if config.verbose:
        # Print some derived quantities.
        print('Telescope effective area: {0:.3f}'
              .format(instrument.effective_area))
        print('Fiber entrance area: {0:.3f}'
              .format(instrument.fiber_area))
        print('Source types: {0}.'.format(instrument.source_types))

    return instrument
