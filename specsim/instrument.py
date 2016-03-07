# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an instrument response for spectroscopic simulations.

An instrument model is usually initialized from a configuration used to create
a simulator and then accessible via its ``instrument`` attribute, for example:

    >>> import specsim.simulator
    >>> simulator = specsim.simulator.Simulator('test')
    >>> print(np.round(simulator.instrument.exposure_time, 1))
    1000.0 s
    >>> simulator.atmosphere.airmass
    1.0

See :doc:`/api` for examples of changing model parameters defined in the
configuration.  Certain parameters can also be changed after a model has
been initialized, for example:

    >>> simulator.instrument.exposure_time = 1200 * u.s

See :class:`Instrument` and :class:`Camera` for details.
"""
from __future__ import print_function, division

import math

import numpy as np
import scipy.sparse

import astropy.constants
import astropy.units as u


class Instrument(object):
    """Model the instrument response of a fiber spectrograph.

    A spectrograph can have multiple cameras with different wavelength
    coverage.

    The only attribute that can be changed after an instrument has been
    created is :attr:`exposure_time`.  File a github issue if you would like
    to expand this list.

    Parameters
    ----------
    name : str
        Descriptive name of this instrument.
    wavelength : nastropy.units.Quantity
        Array of wavelength bin centers where the instrument response is
        calculated, with units.
    fiber_acceptance_dict : dict
        Dictionary of fiber acceptance fractions tabulated for different
        source models, with keys corresponding to source model names.
    cameras : list
        List of :class:`Camera` instances representing the camera(s) of
        this instrument.
    primary_mirror_diameter : astropy.units.Quantity
        Diameter of the primary mirror, with units.
    obscuration_diameter : astropy.units.Quantity
        Diameter of a central obscuration of the primary mirror, with units.
    support_width : astropy.units.Quantity
        Width of the obscuring supports, with units.
    fiber_diameter : astropy.units.Quantity
        Angular field of view diameter of the simulated fibers, with units.
    exposure_time : astropy.units.Quantity
        Exposure time used to scale the instrument response, with units.
    """
    def __init__(self, name, wavelength, fiber_acceptance_dict, cameras,
                 primary_mirror_diameter, obscuration_diameter, support_width,
                 fiber_diameter, exposure_time):
        self.name = name
        self._wavelength = wavelength
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
        energy_per_photon = (hc / self._wavelength).to(u.erg)

        # Calculate the rate of photons incident on the focal plane per
        # wavelength bin per unit spectral flux density. The fiber acceptance
        # fraction is not included in this calculation.
        wavelength_bin_size = np.gradient(self._wavelength)
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
        """Get the tabulated fiber acceptance function for the specified source.

        Parameters
        ----------
        source : specsim.source.Source
            The source whose fiber acceptance should be returned.

        Returns
        -------
        numpy.ndarray
            Array of fiber acceptance values in the range 0-1, tabulated at
            at each :attr:`wavelength`.
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

        Parameters
        ----------
        flux : astropy.units.Quantity
            Constant source flux to use for displaying the instrument response.
        exposure_time : astropy.units.Quantity or None
            Exposure time to use for displaying the instrument response.
            Use the configured exposure time when this parameter is None.
        cmap : str or matplotlib.colors.Colormap
            Matplotlib colormap name or instance to use for displaying the
            instrument response.  Colors are selected for each camera
            according to its central wavelength, so a spectral color map
            will give reasonably intuitive results.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 8))
        ax1_rhs = ax1.twinx()
        ax2_rhs = ax2.twinx()
        cmap = cm.get_cmap(cmap)

        wave = self._wavelength.value
        wave_unit = self._wavelength.unit
        dwave = np.gradient(wave)

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
    """Model the response of a single fiber spectrograph camera.

    No camera attributes can be changed after an instrument has been
    created.  File a github issue if you would like to change this.

    Parameters
    ----------
    name : str
        A brief descriptive name for this camera.  Typically a single letter
        indicating the wavelength band covered by this camera.
    wavelength : astropy.units.Quantity
        Array of wavelength bin centers where the instrument response is
        calculated, with units.  Must be equally spaced.
    throughput : numpy.ndarray
        Array of throughput values tabulated at each wavelength bin center.
    row_size : astropy.units.Quantity
        Array of row size values tabulated at each wavelength bin center.
        Units are required, e.g. Angstrom / pixel.
    fwhm_resolution : astropy.units.Quantity
        Array of wavelength resolution FWHM values tabulated at each wavelength
        bin center. Units are required, e.g., Angstrom.
    neff_spatial : astropy.units.Quantity
        Array of effective trace sizes in the spatial (fiber) direction
        tabulated at each wavelength bin center.  Units are required, e.g.
        pixel.
    read_noise : astropy.units.Quantity
        Camera noise per readout operation.  Units are required, e.g. electron.
    dark_current : astropy.units.Quantity
        Nominal mean dark current from sensor.  Units are required, e.g.
        electron / hour.
    gain : astropy.units.Quantity
        CCD amplifier gain.  Units are required, e.g., electron / adu.
        (This is really 1/gain).
    num_sigmas_clip : float
        Number of sigmas where the resolution should be clipped when building
        a sparse resolution matrix.
    output_pixel_size : astropy.units.Quantity
        Size of output pixels for this camera.  Units are required, e.g.
        Angstrom. Must be a multiple of the the spacing of the wavelength
        input parameter.
    """
    def __init__(self, name, wavelength, throughput, row_size,
                 fwhm_resolution, neff_spatial, read_noise, dark_current,
                 gain, num_sigmas_clip, output_pixel_size):
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
        ccd_start, ccd_stop = ccd_nonzero[0], ccd_nonzero[-1] + 1
        if (np.any(self._fwhm_resolution[:ccd_start] != 0) or
            np.any(self._fwhm_resolution[ccd_stop:] != 0)):
            raise RuntimeError('Resolution extends beyond CCD coverage.')
        if (np.any(self._neff_spatial[:ccd_start] != 0) or
            np.any(self._neff_spatial[ccd_stop:] != 0)):
            raise RuntimeError('Spatial Neff extends beyond CCD coverage.')

        # CCD properties must be valid across the coverage.
        if np.any(self._row_size[ccd_start:ccd_stop] <= 0.):
            raise RuntimeError('CCD row size has invalid values <= 0.')
        if np.any(self._fwhm_resolution[ccd_start:ccd_stop] <= 0.):
            raise RuntimeError('CCD resolution has invalid values <= 0.')
        if np.any(self._neff_spatial[ccd_start:ccd_stop] <= 0.):
            raise RuntimeError('CCD spatial Neff has invalid values <= 0.')

        self.ccd_slice = slice(ccd_start, ccd_stop)
        self.ccd_coverage = np.zeros_like(self._wavelength, dtype=bool)
        self.ccd_coverage[ccd_start:ccd_stop] = True
        self._wavelength_min = self._wavelength[ccd_start]
        self._wavelength_max = self._wavelength[ccd_stop - 1]

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
            self.read_noise * np.sqrt(self.neff_pixels.value) * u.pixel ** 2
            ).to(u.electron)

        # Calculate the dark current per wavelength bin.
        self.dark_current_per_bin = (
            self.dark_current * self.neff_pixels).to(u.electron / u.s)

        # Calculate the RMS resolution assuming a Gaussian PSF.
        fwhm_to_sigma = 1. / (2 * math.sqrt(2 * math.log(2)))
        self._rms_resolution = fwhm_to_sigma * self._fwhm_resolution

        # Find the minimum wavelength that can disperse into the CCD,
        # assuming a constant extrapolation of the resolution.
        sigma_lo = self._rms_resolution[ccd_start]
        min_wave = (self._wavelength[ccd_start] -
                    self.num_sigmas_clip * sigma_lo)
        if min_wave < self._wavelength[0]:
            raise RuntimeError(
                'Wavelength grid min does not cover {0}-camera response.'
                .format(self.name))
        matrix_start = np.where(self._wavelength >= min_wave)[0][0]

        # Find the maximum wavelength that can disperse into the CCD,
        # assuming a constant extrapolation of the resolution.
        sigma_hi = self._rms_resolution[ccd_stop - 1]
        max_wave = (self._wavelength[ccd_stop - 1] +
                    self.num_sigmas_clip * sigma_hi)
        if max_wave > self._wavelength[-1]:
            raise RuntimeError(
                'Wavelength grid max does not cover {0}-camera response.'
                .format(self.name))
        matrix_stop = np.where(self._wavelength <= max_wave)[0][-1] + 1
        self.response_slice = slice(matrix_start, matrix_stop)

        # Pad the RMS array to cover the full resolution matrix range.
        sigma = np.empty((matrix_stop - matrix_start))
        sigma[:ccd_start - matrix_start] = sigma_lo
        sigma[ccd_start - matrix_start:ccd_stop - matrix_start] = (
            self._rms_resolution[ccd_start:ccd_stop])
        sigma[ccd_stop - matrix_start:] = sigma_hi

        # Calculate the range of wavelengths where the dispersion will
        # be evaluated.  The evaluation range extends beyond wavelengths that
        # can disperse into the CCD in order to calculate the normalization.
        wave = self._wavelength[matrix_start:matrix_stop]
        min_wave = wave - self.num_sigmas_clip * sigma
        max_wave = wave + self.num_sigmas_clip * sigma
        eval_start = np.searchsorted(self._wavelength, min_wave)
        eval_stop = np.searchsorted(self._wavelength, max_wave) + 1

        # The columns of the resolution matrix are clipped to the CCD coverage.
        column_start = np.maximum(eval_start, ccd_start)
        column_stop = np.minimum(eval_stop, ccd_stop)
        column_size = column_stop - column_start
        assert np.all(column_size > 0)

        # Prepare start, stop values for slicing eval -> column.
        trim_start = column_start - eval_start
        trim_stop = column_stop - eval_start
        assert np.all(trim_stop > trim_start)

        # Prepare a sparse resolution matrix in compressed column format.
        matrix_size = np.sum(column_size)
        data = np.empty((matrix_size,), float)
        indices = np.empty((matrix_size,), int)
        indptr = np.empty((len(column_size) + 1,), int)
        indptr[0] = 0
        indptr[1:] = np.cumsum(column_size)
        assert indptr[-1] == matrix_size

        # Fill sparse matrix arrays.
        sparse_start = 0
        for i in xrange(matrix_stop - matrix_start):
            eval_slice = slice(eval_start[i], eval_stop[i])
            w = self._wavelength[eval_slice]
            dw = self._wavelength_bin_size[eval_slice]
            column = dw * np.exp(-0.5 * ((w - wave[i]) / sigma[i]) ** 2)
            # Normalize over the full evaluation range.
            column /= np.sum(column)
            # Trim to the CCD coverage.
            s = slice(sparse_start, sparse_start + column_size[i])
            data[s] = column[trim_start[i]:trim_stop[i]]
            indices[s] = np.arange(column_start[i], column_stop[i]) - ccd_start
            sparse_start = s.stop
        assert np.all((indices >= 0) & (indices < ccd_stop - ccd_start))
        assert s.stop == matrix_size

        # Create the matrix in CSC format.
        matrix_shape = (ccd_stop - ccd_start, matrix_stop - matrix_start)
        self._resolution_matrix = scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=matrix_shape)
        # Convert to CSR format for faster matrix multiplies.
        self._resolution_matrix = self._resolution_matrix.tocsr()

        # Initialize downsampled output pixels.
        self._output_pixel_size = (
            output_pixel_size.to(self._wavelength_unit).value)
        # Check that we can downsample simulation pixels to obtain
        # output pixels.  This check will only work if the simulation
        # grid is equally spaced, but no other part of the Camera class
        # class currently requires this.
        wavelength_step = self._wavelength[1] - self._wavelength[0]
        self._downsampling = int(round(
            self._output_pixel_size / wavelength_step))
        num_downsampled = int(
            (self._wavelength_max - self._wavelength_min) //
            self._output_pixel_size)
        pixel_edges = (
            self._wavelength_min - 0.5 * wavelength_step +
            np.arange(num_downsampled + 1) * self._output_pixel_size)
        sim_edges = (
            self._wavelength[self.ccd_slice][::self._downsampling] -
             0.5 * wavelength_step)
        if not np.allclose(
            pixel_edges, sim_edges, rtol=0., atol=1e-6 * wavelength_step):
            raise ValueError(
                'Cannot downsample {0}-camera pixels from {1:f} to {2} {3}.'
                .format(self.name, wavelength_step, self._output_pixel_size,
                        self._wavelength_unit))
        # Save the centers of each output pixel.
        self._output_wavelength = 0.5 * (pixel_edges[1:] + pixel_edges[:-1])
        # Initialize the parameters used by the downsample() method.
        self._output_slice = slice(
            self.ccd_slice.start,
            self.ccd_slice.start + num_downsampled * self._downsampling)
        self._downsampled_shape = (num_downsampled, self._downsampling)


    def get_output_resolution_matrix(self):
        """Return the output resolution matrix.

        The output resolution is calculated by summing output pixel
        blocks of the full resolution matrix.  This is equivalent to
        the convolution of our resolution with a boxcar representing
        an output pixel.

        This operation is relatively slow and requires a lot of memory
        since the full resolution matrix is expanded to a dense array
        during the calculation.

        The result is returned as a dense matrix but will generally be
        sparse, so can be converted to one of the scipy.sparse formats.
        The result is not saved internally.

        Edge effects are not handled very gracefully in order to return
        a square matrix.

        Returns
        -------
        numpy.ndarray
            Square array of resolution matrix elements.
        """
        n = len(self._output_wavelength)
        m = self._downsampling
        i0 = self.ccd_slice.start - self.response_slice.start
        return (self._resolution_matrix[: n * m, i0 : i0 + n * m].toarray()
                .reshape(n, m, n, m).sum(axis=3).sum(axis=1) / float(m))


    def downsample(self, data, method=np.sum):
        """Downsample data tabulated on the simulation grid to output pixels.
        """
        data = np.asanyarray(data)
        if data.shape != self._wavelength.shape:
            raise ValueError(
                'Invalid data shape for downsampling: {0}.'.format(data.shape))

        return method(
            data[self._output_slice].reshape(self._downsampled_shape), axis=-1)


    def apply_resolution(self, flux):
        """
        Input should be on the simulation wavelength grid.

        Any throughput should already be applied.
        """
        flux = np.asarray(flux)
        dispersed = np.zeros_like(flux)

        dispersed[self.ccd_slice] = self._resolution_matrix.dot(
            flux[self.response_slice])

        return dispersed


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


    @property
    def output_pixel_size(self):
        """Size of output pixels.

        Must be a multiple of the simulation wavelength grid.
        """
        return self._output_pixel_size * self._wavelength_unit


    @property
    def output_wavelength(self):
        """Output pixel central wavelengths.
        """
        return self._output_wavelength * self._wavelength_unit


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
            ['read_noise', 'dark_current', 'gain', 'num_sigmas_clip',
             'output_pixel_size'])
        initialized_cameras.append(Camera(
            camera_name, config.wavelength, throughput,
            ccd['row_size'], ccd['fwhm_resolution'],
            ccd['neff_spatial'], constants['read_noise'],
            constants['dark_current'], constants['gain'],
            constants['num_sigmas_clip'], constants['output_pixel_size']))

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
