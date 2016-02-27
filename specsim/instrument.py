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
import collections

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

        for source_type in self.source_types:
            # Plot fiber acceptance fractions without labels.
            ax1.plot(wave, self.fiber_acceptance_dict[source_type], 'k--')
        for camera in self.cameras:
            cwave = camera._wavelength

            # Use an approximate spectral color for each band.
            mid_wave = 0.5 * (camera.wavelength_min + camera.wavelength_max)
            color = cmap(
                (mid_wave - self.wavelength_min) /
                (self.wavelength_max - self.wavelength_min))

            # Calculate number of photons with perfect fiber acceptance.
            nphot = (flux * self.photons_per_bin * exposure_time *
                     camera.throughput / dwave)
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

            ax2_rhs.plot(
                cwave, camera.neff_spatial.to(u.pixel), ls=':', color=color)

        ax1.plot([], [], 'k--', label='Fiber Acceptance')
        ax1.plot([], [], 'k-', label='Camera Throughput')
        ax1.plot([], [], 'k:', label='{0}'.format(flux))
        ax1.plot([], [], 'k-.', label='Dark + Readout Noise')
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

    Parameters
    ----------
    """
    def __init__(self, name, wavelength, throughput, row_size,
                 fwhm_resolution, neff_spatial, read_noise, dark_current,
                 gain, num_sigmas_clip):
        self.name = name
        self._wavelength = wavelength.to(self._wavelength_unit).value
        self.throughput = throughput
        self._row_size = row_size.to(self._wavelength_unit / u.pixel).value
        self._fwhm_resolution = fwhm_resolution.to(self._wavelength_unit).value
        self._neff_spatial = neff_spatial.to(u.pixel).value
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.gain = gain
        self.num_sigmas_clip = num_sigmas_clip

        # The arrays defining the CCD properties must all have identical
        # wavelength coverage.
        ccd_nonzero = np.where(self._row_size > 0)[0]
        ccd_first, ccd_last = ccd_nonzero[0], ccd_nonzero[-1] + 1
        if (np.any(self._fwhm_resolution[:ccd_first] != 0) or
            np.any(self._fwhm_resolution[ccd_last:] != 0)):
            raise RuntimeError('Resolution extends beyond CCD coverage.')
        if (np.any(self._neff_spatial[:ccd_first] != 0) or
            np.any(self._neff_spatial[ccd_last:] != 0)):
            raise RuntimeError('Spatial Neff extends beyond CCD coverage.')

        # CCD properties must be valid across the coverage.
        if np.any(self._row_size[ccd_first:ccd_last] <= 0.):
            raise RuntimeError('CCD row size has invalid values <= 0.')
        if np.any(self._fwhm_resolution[ccd_first:ccd_last] <= 0.):
            raise RuntimeError('CCD resolution has invalid values <= 0.')
        if np.any(self._neff_spatial[ccd_first:ccd_last] <= 0.):
            raise RuntimeError('CCD spatial Neff has invalid values <= 0.')

        self.ccd_slice = slice(ccd_first, ccd_last)
        self.ccd_coverage = np.zeros_like(self._wavelength, dtype=bool)
        self.ccd_coverage[ccd_first:ccd_last] = True
        self._wavelength_min = self._wavelength[ccd_first]
        self._wavelength_max = self._wavelength[ccd_last - 1]

        # Calculate the size of each wavelength bin in units of pixel rows.
        self._wavelength_bin_size = np.gradient(self._wavelength)
        neff_wavelength = np.zeros_like(self._neff_spatial)
        neff_wavelength[self.ccd_slice] = (
            self._wavelength_bin_size[self.ccd_slice] /
            self._row_size[self.ccd_slice])

        # Calculate the effective pixel area contributing to the signal
        # in each wavelength bin.
        self.neff_pixels = neff_wavelength * self._neff_spatial * u.pixel ** 2

        # Calculate the read noise per wavelength bin, assuming that
        # readnoise is uncorrelated between pixels (hence the sqrt scaling). The
        # value will be zero in pixels that are not used by this camera.
        self.read_noise_per_bin = (
            self.read_noise * np.sqrt(self.neff_pixels.value)
            ).to(u.electron)

        # Calculate the dark current per wavelength bin.
        self.dark_current_per_bin = (
            self.dark_current * self.neff_pixels).to(u.electron / u.s)

        # Calculate the RMS resolution assuming a Gaussian PSF.
        fwhm_to_sigma = 1. / (2 * math.sqrt(2 * math.log(2)))
        self._rms_resolution = fwhm_to_sigma * self._fwhm_resolution

        return
        # Build a resolution matrix that transforms source flux on a true
        # wavelength grid to flux on an observed wavelength grid.
        # The matrix is not square because source flux can disperse into
        # the CCD from just outside its wavelength limits.
        columns = collections.deque()
        # Add wavelengths below the CCD coverage that can disperse into it.
        print('=== under')
        response_start = self.ccd_slice.start
        try:
            while True:
                columns.appendleft(self._resolution_column(response_start - 1))
                response_start -= 1
        except IndexError:
            # Flux centered at bin_index cannot disperse into the CCD.
            pass
        # Add wavelengths covered by the CCD.
        print('=== ccd')
        for bin_index in xrange(self.ccd_slice.start, self.ccd_slice.stop):
            columns.append(self._resolution_column(bin_index))
        # Add wavelengths above the CCD coverage that can disperse into it.
        print('=== over')
        response_stop = self.ccd_slice.stop
        try:
            while True:
                columns.append(self._resolution_column(response_stop))
                response_stop += 1
        except IndexError:
            # Flux centered at bin_index cannot disperse into the CCD.
            pass
        self.response_slice = slice(response_start, response_stop)
        print('response:', self.response_slice)


    # Canonical wavelength unit used for all internal arrays.
    _wavelength_unit = u.Angstrom


    @property
    def wavelength_min(self):
        """Minimum wavelength covered by this camera's CCD.
        """
        return self._wavelength_min * self._wavelength_unit


    @property
    def wavelength_max(self):
        """Maximum wavelength covered by this camera's CCD.
        """
        return self._wavelength_max * self._wavelength_unit


    @property
    def rms_resolution(self):
        """Array of RMS resolution values.
        """
        return self._rms_resolution * self._wavelength_unit


    @property
    def row_size(self):
        """Array of row sizes in the dispersion direction.
        """
        return self._row_size * self._wavelength_unit / u.pixel


    @property
    def neff_spatial(self):
        """Array of effective pixel dimensions in the spatial (fiber) direction.
        """
        return self._neff_spatial * u.pixel


    def _resolution_column(self, bin_index):
        """Evaluate a single column of our resolution matrix.

        A column gives the detector response to a delta-function input centered
        in the specified bin, in terms of observed wavelengths.

        The resolution is modeled as a Gaussian clipped and renormalized to
        num_sigmas_clip.
        """
        # Determine the RMS resolution to use, with constant extrapolation
        # of the resolution below and above the CCD.
        if bin_index < self.ccd_slice.start:
            sigma = self._rms_resolution[self.ccd_slice.start]
        elif bin_index >= self.ccd_slice.stop:
            sigma = self._rms_resolution[self.ccd_slice.stop - 1]
        else:
            sigma = self._rms_resolution[bin_index]
        assert sigma > 0

        # Calculate the non-zero extent of this column.
        dwave = self.num_sigmas_clip * sigma
        bin_wave = self._wavelength[bin_index]
        min_wave = bin_wave - dwave
        max_wave = bin_wave + dwave

        # Is the wavelength grid big enough?
        if min_wave < self._wavelength[0]:
            raise RuntimeError(
                'Wavelength grid min does not cover {0}-camera response.'
                .format(self.name))
        if max_wave > self._wavelength[-1]:
            raise RuntimeError(
                'Wavelength grid max does not cover {0}-camera response.'
                .format(self.name))

        # Does this column's response overlap the CCD?
        start = np.where(self._wavelength <= min_wave)[0][-1]
        stop = np.where(self._wavelength >= max_wave)[0][0] + 1
        if stop <= self.ccd_slice.start or start >= self.ccd_slice.stop:
            raise IndexError('Column does not overlap CCD.')

        # Calculate the clipped and renormalized dispersion in each bin.
        wave = self._wavelength[start:stop]
        dwave = self._wavelength_bin_size[start:stop]
        column = np.exp(-0.5 * (wave / sigma) ** 2) * dwave
        column /= np.sum(column)

        # Trim the column to the CCD wavelength coverage.
        if start < self.ccd_slice.start:
            column = column[self.ccd_slice.start - start:]
            start = self.ccd_slice.start
        if stop > self.ccd_slice.stop:
            column = column[:self.ccd_slice.stop - stop]
            stop = self.ccd_slice.stop

        return column, (start, stop)


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
            ['read_noise', 'dark_current', 'gain', 'num_sigmas_clip'])
        initialized_cameras.append(Camera(
            camera_name, config.wavelength, throughput,
            ccd['row_size'], ccd['fwhm_resolution'],
            ccd['neff_spatial'], constants['read_noise'],
            constants['dark_current'], constants['gain'],
            constants['num_sigmas_clip']))

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
