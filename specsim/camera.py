# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model a fiber spectrograph camera for spectroscopic simulations.

Cameras belong to an :class:`Instrument <specsim.instrument.Instrument>` and
are usually initialized from a configuration used to create
a simulator and then accessible via its ``instrument.cameras`` attribute,
for example:

    >>> import specsim.simulator
    >>> simulator = specsim.simulator.Simulator('test')  # doctest: +IGNORE_OUTPUT
    >>> print(np.round(simulator.instrument.cameras[0].read_noise, 1))
    2.9 electron / pix2

See :doc:`/api` for examples of changing model parameters defined in the
configuration. No attributes can be changed after a simulator has
been created.  File a github issue if you would like to change this.
"""
from __future__ import print_function, division

import numpy as np
import scipy.sparse

import astropy.units as u


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
    allow_convolution : bool
        Set True to precompute the sparse resolution matrix needed by
        :meth:`get_output_resolution_matrix`, :meth:`apply_resolution` and
        :meth:`downsample`.
    """
    def __init__(self, name, wavelength, throughput, row_size,
                 fwhm_resolution, neff_spatial, read_noise, dark_current,
                 gain, num_sigmas_clip, output_pixel_size,
                 allow_convolution=True):
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
        self.allow_convolution = allow_convolution

        # The arrays defining the CCD properties must all have identical
        # wavelength coverage.
        ccd_nonzero = np.where(np.logical_and(self._row_size > 0, self.throughput != 0))[0]
        #ccd_nonzero = np.where(self._row_size > 0)[0]
        ccd_start, ccd_stop = ccd_nonzero[0], ccd_nonzero[-1] + 1 
        #if (np.any(self._fwhm_resolution[:ccd_start] != 0) or
         #   np.any(self._fwhm_resolution[ccd_stop:] != 0)):
          #  raise RuntimeError('Resolution extends beyond CCD coverage.')
        #if (np.any(self._neff_spatial[:ccd_start] != 0) or
         #   np.any(self._neff_spatial[ccd_stop:] != 0)):
          #  raise RuntimeError('Spatial Neff extends beyond CCD coverage.')

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
        fwhm_to_sigma = 1. / (2 * np.sqrt(2 * np.log(2)))
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

        # The remaining steps are only necessary to support convolution
        # and downsampling.
        if not self.allow_convolution:
            return

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
        for i in range(matrix_stop - matrix_start):
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
        if not np.allclose(np.diff(self._wavelength), wavelength_step):
            raise RuntimeError(
                'Non-uniform simulation wavelength grid not supported yet.')
        self._downsampling = int(round(
            self._output_pixel_size / wavelength_step))
        if not np.allclose(self._downsampling * wavelength_step,
                           self._output_pixel_size):
            raise ValueError(
                'Invalid output_pixel_size {0} for {1} camera: '
                .format(output_pixel_size, self.name) +
                'must be multiple of {0} {1}.'
                .format(wavelength_step, self._wavelength_unit))
        # The self._wavelength array stores the centers of fixed-width bins.
        # Calculate the edges of the downsampled output pixels. Trim
        # any partial output pixel on the high end.
        output_min = self._wavelength_min - 0.5 * wavelength_step
        output_max = self._wavelength_max + 0.5 * wavelength_step
        num_downsampled = int(np.floor(
            (output_max - output_min) / self._output_pixel_size))
        pixel_edges = (
            output_min +
            np.arange(num_downsampled + 1) * self._output_pixel_size)
        # Save the centers of each output pixel.
        self._output_wavelength = 0.5 * (pixel_edges[1:] + pixel_edges[:-1])
        # Initialize the parameters used by the downsample() method.
        self._output_slice = slice(
            self.ccd_slice.start,
            self.ccd_slice.start + num_downsampled * self._downsampling)
        self._downsampled_shape = (num_downsampled, self._downsampling)

    def get_output_resolution_matrix(self):
        """Return the output resolution matrix in DIA sparse format.

        The output resolution is calculated by summing output pixel
        blocks of the full resolution matrix.  This is equivalent to
        the convolution of our resolution with a boxcar representing
        an output pixel. Edge effects are not handled very gracefully
        in order to return a square matrix.

        The memory required for this operation scales with the number
        of non-zero elements in the returned matrix. This matrix is
        not used internally and is re-calcuated each time this method
        is called.

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Square array of resolution matrix elements in the DIA
            sparse format.
        """
        if not self.allow_convolution:
            raise RuntimeError('Camera created with allow_convolution False.')
        n = len(self._output_wavelength)
        m = self._downsampling
        i0 = self.ccd_slice.start - self.response_slice.start
        output_slice = slice(i0, i0 + n * m)
        # Initialize CSR format arrays for building the output matrix.
        indptr_out = np.empty((n + 1,), int)
        indices_out = []
        data_out = []
        row_size = self._resolution_matrix.shape[1]
        cols_sum = np.empty(row_size, int)
        data_sum = np.empty(row_size, float)
        # Loop over rows of the CSR format sparse data.
        indices_in = self._resolution_matrix.indices
        indptr_in = self._resolution_matrix.indptr
        data_in = self._resolution_matrix.data
        # Loop over rows in the full resolution matrix.
        num_out = 0
        for i in range(0, n * m, m):
            cols_sum[:] = 0
            data_sum[:] = 0.
            # Loop over rows that will be combined into a single output row.
            for k in range(i, i + m):
                packed = slice(indptr_in[k], indptr_in[k + 1])
                # Find the columns with data in this row.
                expanded = indices_in[packed]
                # Count the rows contributing to each column.
                cols_sum[expanded] += 1
                # Sum the data across rows for each column.
                data_sum[expanded] += data_in[packed]
            # Combine into a single output row.
            data = data_sum[output_slice].reshape(n, m).sum(axis=1) / m
            counts = cols_sum[output_slice].reshape(n, m).sum(axis=1)
            indices = np.where(counts > 0)[0]
            indices_out.append(indices)
            data_out.append(data[indices])
            indptr_out[i // m] = num_out
            num_out += len(indices)
            indptr_out[(i // m) + 1] = num_out
        # Combine row arrays.
        data_out = np.hstack(data_out)
        indices_out = np.hstack(indices_out)

        # Build the output matrix in CSR format.
        R = scipy.sparse.csr_matrix((data_out, indices_out, indptr_out), (n, n))

        # Convert to DIA format and return.
        return R.todia()

    def centers_to_edges(self, centers, kind='cubic'):
        """Calculate bin edges from bin centers.
        Edges are calculated with interpolation (or extrapolation on the edges)
        from integer to half-integer indices.
        Parameters
        ----------
        centers : array
            1D array of N increasing center values with at least 2 values.
        kind : str or int
            Passed to :func:`scipy.interpolate.interp1d`. When N < 4,
            'linear' is always used.
        Returns
        -------
        array
            1D array of N+1 increasing bin edge values.
        """
        centers = np.asarray(centers)
        if len(centers.shape) != 1:
            raise ValueError('Expected 1D array of centers.')
        if len(centers) < 2:
            raise ValueError('Need at least 2 centers.')
        elif len(centers) < 4:
            kind = 'linear'
        if not np.all(np.diff(centers) > 0):
            raise ValueError('Expected increasing center values.')

        center_idx = np.arange(len(centers))
        interpolator = scipy.interpolate.interp1d(
            center_idx, centers, fill_value='extrapolate', copy=False,
            assume_sorted=True, kind=kind)

        edge_idx = np.arange(len(centers) + 1.) - 0.5
        return interpolator(edge_idx)

    def resample_binned(self, edges_out, edges_in, hist_in, axis=-1, zero_pad=True):
        """Flux conserving linear resampler of binned data.
        Parameters
        ----------
        edges_out : array
            1D array of M >= 2 output bin edges, in increasing order.
        edges_in : array
            1D array of N >= 2 input bin edges, in increasing order.
        hist_in : array
            Array with N-1 input bin values indexed by the specified axis.
        axis : int
            Axis of hist_in used to index input histogram values.
        zero_pad : bool
            When True, allow the output edges to extend beyond the input
            edges and assume that the input histogram is zero outside of
            its extent.  When False, raises a ValueError if extrapolation
            would be required.
        Returns
        -------
        array
            1D array of M-1 resampled bin values.
        """
        # Check input 1D arrays.
        edges_out = np.asarray(edges_out)
        edges_in = np.asarray(edges_in)
        if edges_out.ndim != 1:
            raise ValueError('Expected 1D edges_out.')
        if edges_in.ndim != 1:
            raise ValueError('Expected 1D edges_in.')
        if len(edges_out) < 2:
            raise ValueError('Need at least one output bin.')
        if len(edges_in) < 2:
            raise ValueError('Need at least one input bin.')
        binsize_out = np.diff(edges_out)
        if np.any(binsize_out <= 0):
            raise ValueError('Expecting increasing edges_out.')
        binsize_in = np.diff(edges_in)
        if np.any(binsize_in <= 0):
            raise ValueError('Expecting increasing edges_in.')
        if not zero_pad and ((edges_out[0] < edges_in[0]) or
                            (edges_out[-1] > edges_in[-1])):
            raise ValueError(
                'Ouput bins extend beyond input bins but zero_pad is False.')
        if (edges_out[0] >= edges_in[-1]) or (edges_out[-1] <= edges_in[0]):
            raise ValueError('Input and output bins do not overlap.')
        # Check input histogram(s) and put index in axis 0. Note that we
        # use moveaxis instead of rollaxis since it is easier to invert.
        hist_in = np.asarray(hist_in)
        hist_in_rolled = np.moveaxis(hist_in, axis, 0)
        if len(hist_in_rolled) != len(edges_in) - 1:
            raise ValueError(
                'Unexpected length of hist_in along axis {}.'.format(axis))
        # Align output edges to input edges.
        idx = np.searchsorted(edges_in, edges_out)
        # Loop over output bins.
        nin = len(edges_in) - 1
        nout = len(edges_out) - 1
        hist_out_rolled_shape = (nout,) + hist_in_rolled.shape[1:]
        hist_out_rolled = np.zeros(hist_out_rolled_shape)
        hi = idx[0]
        for i in range(nout):
            lo = hi
            hi = idx[i + 1]
            if (lo > nin) or (hi == 0):
                # This bin does not overlap the input.
                continue
            if lo == hi:
                # Output bin is fully embedded within an input bin:
                # give it a linear share.
                hist_out_rolled[i] = (
                    binsize_out[i] / binsize_in[lo - 1] * hist_in_rolled[lo - 1])
                continue
            # Calculate fraction of first input bin overlapping this output bin.
            if lo > 0:
                hist_out_rolled[i] += (
                    hist_in_rolled[lo - 1] / binsize_in[lo - 1] * (
                        edges_in[lo] - edges_out[i]))
            # Calculate fraction of last input bin overlaping this output bin.
            if hi <= nin:
                hist_out_rolled[i] += (
                    hist_in_rolled[hi - 1] / binsize_in[hi - 1] * (
                        edges_out[i + 1] - edges_in[hi - 1]))
            # Add input bins fully contained within this output bin.
            if hi > lo + 1:
                hist_out_rolled[i] += np.sum(hist_in_rolled[lo:hi - 1], axis=0)
        # Return our unrolled output histogram.
        return np.moveaxis(hist_out_rolled, axis, 0)

    def eboss_wave_out(self, x_in):
        return(10 ** np.arange(np.log10(x_in[0]), np.log10(x_in[-1]), 0.0001))

    def downsample_to_eboss(self, x_out, x_in, y_in, axis=0, zero_pad=True):
        """Flux conserving linear resampling of density samples.
        By "density" we mean that the integral of y(x) is the conserved flux.
        This function is just a wrapper around :func:`resample_binned` that:
        - Estimates input bin edges.
        - Multiplies each density y(x[i]) by the bin width to obtain bin contents.
        - Resamples the binned data.
        - Divides output bin values by output bin widths to obtain densities.
        The special case of a single output sample location is handled with linear
        interpolation of the input densities, so is not really flux conserving but
        probably what you want in this case.
        Parameters
        ----------
        x_out : array
            1D array of M >= 1 output sample locations.
        x_in : array
            1D array of N >= 2 input sample locations.
        y_in : array
            Array of N input sample densities indexed by the specified axis.
        axis : int
            Axis of y_in used to index input density values.
        zero_pad : bool
            When True, allow the output edges to extend beyond the input
            edges and assume that the input histogram is zero outside of
            its extent.  When False, raises a ValueError if extrapolation
            would be required.
        Returns
        -------
        array
            1D array of M output densities.
        """
        x_in = np.asarray(x_in)
        x_out = np.asarray(x_out)
        y_in = np.asarray(y_in)
        if len(x_out.shape) == 0 or len(x_out) == 1:
            # Resampling to a single value.
            if not zero_pad and ((x_out < np.min(x_in)) or (x_out > np.max(x_in))):
                raise ValueError(
                    'Cannot resample outside range when zero_pad is False.')
            # Linearly interpolate x_out in (x_in, y_in).
            return np.interp(x_out, x_in, y_in, left=0., right=0.)
        edges_out = self.centers_to_edges(x_out)
        edges_in = self.centers_to_edges(x_in)
        dx_shape = np.ones(y_in.ndim, int)
        dx_shape[axis] = -1
        hist_in = y_in * np.diff(edges_in).reshape(dx_shape)
        hist_out = self.resample_binned(edges_out, edges_in, hist_in, axis, zero_pad)
        return hist_out / np.diff(edges_out).reshape(dx_shape)

    def downsample(self, data, method=np.sum):
        """Downsample data tabulated on the simulation grid to output pixels.
        """
        if not self.allow_convolution:
            raise RuntimeError('Camera created with allow_convolution False.')
        data = np.asanyarray(data)
        if data.shape[0] != len(self._wavelength):
            raise ValueError(
                'Invalid data shape for downsampling: {0}.'.format(data.shape))

        output = data[self._output_slice]
        new_shape = self._downsampled_shape + output.shape[1:]
        return method(output.reshape(new_shape), axis=1)

    def apply_resolution(self, flux):
        """
        Input should be on the simulation wavelength grid.

        Any throughput should already be applied.
        """
        if not self.allow_convolution:
            raise RuntimeError('Camera created with allow_convolution False.')
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
        if not self.allow_convolution:
            raise RuntimeError('Camera created with allow_convolution False.')
        return self._output_pixel_size * self._wavelength_unit

    @property
    def output_wavelength(self):
        """Output pixel central wavelengths.
        """
        if not self.allow_convolution:
            raise RuntimeError('Camera created with allow_convolution False.')
        return self._output_wavelength * self._wavelength_unit
