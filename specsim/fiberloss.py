# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate fiberloss fractions.

Fiberloss fractions are computed as the overlap between the light profile
illuminating a fiber and the on-sky aperture of the fiber.
"""
from __future__ import print_function, division

import numpy as np
import numpy.lib.stride_tricks

import astropy.units as u
import astropy.table


class GalsimFiberlossCalculator(object):
    """
    Initialize a fiberloss calculator that uses GalSim.

    Parameters
    ----------
    fiber_diameter : float
        Fiber diameter in microns.
    wlen_grid : array
        Array of wavelengths in Angstroms where fiberloss will be calculated.
    num_pixels : int
        Number of pixels to cover the fiber aperture along each axis.
    oversampling : int
        Oversampling factor for anti-aliasing the circular fiber aperture.
    moffat_beta : float
        Beta parameter value for the atmospheric PSF Moffat profile.
    maximum_fft_size : int
        Maximum size of FFT allowed.
    """
    def __init__(self, fiber_diameter, wlen_grid, num_pixels=32,
                 oversampling=16, moffat_beta=3.5, maximum_fft_size=32767):

        self.wlen_grid = np.asarray(wlen_grid)
        self.moffat_beta = moffat_beta

        # Defer import to runtime.
        import galsim

        # Prepare an image of the fiber aperture for numerical integration.
        # Images are formed in the physical x-y space of the focal plane
        # rather than on-sky angles.
        scale = fiber_diameter / num_pixels
        self.image = galsim.Image(num_pixels, num_pixels, scale=scale)

        self.gsparams = galsim.GSParams(maximum_fft_size=32767)

        # Prepare an anti-aliased image of the fiber aperture.
        nos = num_pixels * oversampling
        dxy = (np.arange(nos) + 0.5 - 0.5 * nos) / (0.5 * nos)
        rsq = dxy ** 2 + dxy[:, np.newaxis] ** 2
        inside = (rsq <= 1).astype(float)
        s0, s1 = inside.strides
        blocks = numpy.lib.stride_tricks.as_strided(
            inside, shape=(num_pixels, num_pixels, oversampling, oversampling),
            strides=(oversampling * s0, oversampling * s1, s0, s1))
        self.aperture = blocks.sum(axis=(2, 3)) / oversampling ** 2


    def create_source(self, fractions, half_light_radius,
                      minor_major_axis_ratio, position_angle):
        """Create a model for the on-sky profile of a single source.

        Size and shape parameter values for any component that is not
        present (because its fraction is zero) are ignored.

        Parameters
        ----------
        fractions : array
            Array of length 2 giving the disk and bulge fractions, respectively,
            which must be in the range [0,1] (but this is not checked). If
            their sum is less than one, the remainder is modeled as a point-like
            component.
        half_light_radius : array
            Array of length 2 giving the disk and bulge half-light radii in
            arcseconds, respectively.
        minor_major_axis_ratio : array
            Array of length 2 giving the dimensionless on-sky ellipse
            minor / major axis ratio for the disk and bulge components,
            respectively.
        position_angle : array
            Array of length 2 giving the position angle in degrees of the on-sky
            disk and bluge ellipses, respectively.  Angles are measured counter
            clockwise relative to the +x axis.

        Returns
        -------
        galsim.GSObject
            A object representing the sum of all requested components with its
            total flux normalized to one.
        """
        # This is a no-op but still required to define the namespace.
        import galsim

        components = []
        if fractions[0] > 0:
            # Disk component
            components.append(galsim.Exponential(
                flux=fractions[0], half_light_radius=half_light_radius[0])
                .shear(q=minor_major_axis_ratio[0],
                       beta=position_angle[0] * galsim.degrees))
        if fractions[1] > 0:
            components.append(galsim.DeVaucouleurs(
                flux=fractions[1], half_light_radius=half_light_radius[1])
                .shear(q=minor_major_axis_ratio[1],
                       beta=position_angle[1] * galsim.degrees))
        star_fraction = 1 - fractions.sum()
        if star_fraction > 0:
            # Model a point-like source with a tiny (0.001 arcsec) Gaussian.
            # TODO: sigma should be in arcsec here, not microns!
            components.append(galsim.Gaussian(
                flux=star_fraction, sigma=1e-3 * self.image.scale))
        # Combine the components and transform to focal-plane microns.
        return galsim.Add(components, gsparams=self.gsparams)


    def calculate(self, seeing_fwhm, scale, offset, blur_rms,
                  source_fraction, source_half_light_radius,
                  source_minor_major_axis_ratio, source_position_angle,
                  saved_images_file=None):
        """Calculate the acceptance fractions for a set of fibers.

        Parameters
        ----------
        seeing_fwhm : array
            Array of length num_wlen giving the FWHM seeing in arcseconds
            at each wavelength.
        scale : array
            Array of shape (num_fibers, 2) giving the x and y image scales in
            microns / arcsec at each fiber location.
        offset : array
            Array of shape (num_fibers, num_wlen, 2) giving the x and y offsets
            in microns at each fiber location and wavelength.
        blur_rms : array
            Array of shape (num_fibers, num_wlen) giving the RMS instrumental
            Gaussian blur at each fiber location and wavelength.
        source_fraction : array
            Array of shape (num_fibers, 2).  See :meth:`create_source`
            for details.
        source_half_light_radius : array
            Array of shape (num_fibers, 2).  See :meth:`create_source`
            for details.
        source_minor_major_axis_ratio : array
            Array of shape (num_fibers, 2).  See :meth:`create_source`
            for details.
        source_position_angle : array
            Array of shape (num_fibers, 2).  See :meth:`create_source`
            for details.
        saved_images_file : str or None
            Write a multi-extension FITS file with this name containing images
            of the atmospheric and instrument PSFs as a function of wavelength,
            as well as the source profile and the anti-aliased fiber aperture.

        Returns
        -------
        array
            Array of fiberloss fractions in the range 0-1 with shape
            (num_fibers, num_wlen).
        """
        # This is a no-op but still required to define the namespace.
        import galsim

        num_fibers, num_wlen = len(offset), len(self.wlen_grid)
        assert seeing_fwhm.shape == (num_wlen,)
        assert scale.shape == (num_fibers, 2)
        assert offset.shape == (num_fibers, num_wlen, 2)
        assert blur_rms.shape == (num_fibers, num_wlen)
        assert source_fraction.shape == (num_fibers, 2)
        assert source_half_light_radius.shape == (num_fibers, 2)
        assert source_minor_major_axis_ratio.shape == (num_fibers, 2)
        assert source_position_angle.shape == (num_fibers, 2)

        assert np.all(source_fraction >= 0) and np.all(source_fraction <= 1)
        star_fraction = 1 - source_fraction.sum(axis=1)
        assert np.all(star_fraction >= 0) and np.all(star_fraction <= 1)

        if saved_images_file is not None:
            import astropy.io.fits
            import astropy.wcs
            hdu_list = astropy.io.fits.HDUList()
            header = astropy.io.fits.Header()
            header['COMMENT'] = 'Fiberloss calculation images.'
            hdu_list.append(astropy.io.fits.PrimaryHDU(header=header))
            # All subsequent HDUs contain images with the same WCS.
            w = astropy.wcs.WCS(naxis=2)
            w.wcs.ctype = ['x', 'y']
            ny, nx = self.image.array.shape
            w.wcs.crpix = [nx / 2. + 0.5, nx / 2. + 0.5]
            w.wcs.cdelt = [self.image.scale, self.image.scale]
            w.wcs.crval = [0., 0.]
            header = w.to_header()
            # Save the anti-aliased fiber aperture.
            header['COMMENT'] = 'Fiber aperture'
            hdu_list.append(astropy.io.fits.ImageHDU(
                data=self.aperture, header=header))

        scaled_offset = offset / self.image.scale
        fiberloss = np.empty((num_fibers, num_wlen))
        source_profiles = []

        for i, wlen in enumerate(self.wlen_grid):
            # Create the atmospheric PSF for this wavelength in
            # on-sky coordinates.
            seeing = galsim.Moffat(fwhm=seeing_fwhm[i], beta=self.moffat_beta)
            # Loop over fibers.
            for j in range(num_fibers):
                # Transform the atmospheric PSF to the focal plane for
                # this fiber location.
                atmospheric_psf = seeing.transform(
                    scale[j, 0], 0, 0, scale[j, 1]).withFlux(1)
                # Create the instrument PSF for this fiber and wavelength.
                instrument_psf = galsim.Gaussian(sigma=blur_rms[j, i])
                if i == 0:
                    # Create the source profile for this fiber on the sky.
                    source_profile = self.create_source(
                        source_fraction[j], source_half_light_radius[j],
                        source_minor_major_axis_ratio[j],
                        source_position_angle[j])
                    # Transform to focal-plane coordinates.
                    source_profile = source_profile.transform(
                        scale[j, 0], 0, 0, scale[j, 1]).withFlux(1)
                    source_profiles.append(source_profile)
                else:
                    # Lookup the source model for this fiber.
                    source_profile = source_profiles[j]
                # Convolve the source + instrument + astmosphere.
                convolved = galsim.Convolve(
                    [instrument_psf, atmospheric_psf, source_profile],
                    gsparams=self.gsparams)
                # Render the convolved model with its offset.
                offsets = (scaled_offset[j, i, 0], scaled_offset[j, i, 1])
                # TODO: compare method='no_pixel' and 'auto' for
                # accuracy and speed.
                draw_args = dict(image=self.image, method='auto')
                convolved.drawImage(offset=offsets, **draw_args)
                # Calculate the fiberloss fraction for this fiber and wlen.
                fiberloss[j, i] = np.sum(self.image.array * self.aperture)

                if saved_images_file is not None:
                    header['FIBER'] = j
                    header['WLEN'] = wlen
                    header['FRAC'] = fiberloss[j, i]
                    header['COMMENT'] = 'Convolved model'
                    hdu_list.append(astropy.io.fits.ImageHDU(
                        data=self.image.array.copy(), header=header))
                    # The component models are only rendered individually if we
                    # need to save them.
                    instrument_psf.drawImage(offset=offsets, **draw_args)
                    header['COMMENT'] = 'Instrument blur model'
                    hdu_list.append(astropy.io.fits.ImageHDU(
                        data=self.image.array.copy(), header=header))
                    # Render the seeing without the instrumental offset.
                    atmospheric_psf.drawImage(**draw_args)
                    header['COMMENT'] = 'Atmospheric seeing model'
                    hdu_list.append(astropy.io.fits.ImageHDU(
                        data=self.image.array.copy(), header=header))
                    if wlen == self.wlen_grid[-1]:
                        # Render the source profile without any offset after
                        # all other postage stamps for this fiber.
                        source_profile.drawImage(**draw_args)
                        del header['WLEN']
                        del header['FRAC']
                        header['COMMENT'] = 'Source profile'
                        hdu_list.append(astropy.io.fits.ImageHDU(
                            data=self.image.array.copy(), header=header))

        if saved_images_file is not None:
            hdu_list.writeto(saved_images_file, clobber=True)

        return fiberloss


def calculate_fiber_acceptance_fraction(
    focal_x, focal_y, wavelength, source, atmosphere, instrument,
    oversampling = 16, saved_images_file=None, saved_table_file=None):
    """Calculate the acceptance fraction for a single fiber.

    The behavior of this function is customized by the instrument.fiberloss
    configuration parameters. When instrument.fiberloss.method == 'table',
    pre-tabulated values are returned using source.type as the key and
    all other parameters to this function are ignored.

    When instrument.fiberloss.method == 'galsim', fiberloss is calculated
    on the fly using the GalSim package via :class:`GalsimFiberlossCalculator`
    to model the PSF components and source profile and perform the convolutions.

    To efficiently calculate fiberloss fractions for multiple sources with
    GalSim, use :class:`GalsimFiberlossCalculator` directly instead of
    repeatedly calling this method.  See :mod:`specsim.quickfiberloss` for an
    example of this approach.

    Parameters
    ----------
    focal_x : :class:`astropy.units.Quantity`
        X coordinate of the fiber center in the focal plane with length units.
    focal_y : :class:`astropy.units.Quantity`
        Y coordinate of the fiber center in the focal plane with length units.
    wavelength : :class:`astropy.table.Column`
        Array of simulation wavelengths where the fiber acceptance fraction
        should be tabulated.
    source : :class:`specsim.source.Source`
        Source model to use for the calculation.
    atmosphere : :class:`specsim.atmosphere.Atmosphere`
        Atmosphere model to use for the calculation.
    instrument : :class:`specsim.instrument.Instrument`
        Instrument model to use for the calculation.
    oversampling : int
        Oversampling factor to use for anti-aliasing the fiber aperture.
    saved_images_file : str or None
        See :meth:`GalsimFiberlossCalculator.calculate`.
    saved_table_file : str or None
        Write a table of calculated values to a file with this name.  The
        extension determines the file format, and .ecsv is recommended.
        The saved file can then be used as a pre-tabulated input with
        instrument.fiberloss.method = 'table'.

    Returns
    -------
    numpy array
        Array of fiber acceptance fractions (dimensionless) at each of the
        input wavelengths.
    """
    # Use pre-tabulated fiberloss vs wavelength when available.
    num_wlen = instrument.fiberloss_num_wlen
    if num_wlen == 0:
        return instrument.fiber_acceptance_dict[source.type_name]

    # Initialize the grid of wavelengths where the fiberloss will be
    # calculated.
    wlen_unit = wavelength.unit
    wlen_grid = np.linspace(wavelength.data[0], wavelength.data[-1],
                            num_wlen) * wlen_unit

    # Initialize a new calculator.
    calc = GalsimFiberlossCalculator(
        instrument.fiber_diameter.to(u.um).value,
        wlen_grid.to(u.Angstrom).value,
        instrument.fiberloss_num_pixels,
        oversampling,
        atmosphere.seeing['moffat_beta'])

    # Calculate the focal-plane optics at the fiber locations.
    scale, blur, offset = instrument.get_focal_plane_optics(
        focal_x.reshape(1,), focal_y.reshape(1,), wlen_grid)

    # Calculate the atmospheric seeing at each wavelength.
    seeing_fwhm = atmosphere.get_seeing_fwhm(wlen_grid).to(u.arcsec).value

    # Lookup the source model parameters, which we assume to be achromatic.
    source_fraction = np.empty((1, 2))
    source_hlr = np.empty((1, 2))
    source_q = np.empty((1, 2))
    source_beta = np.empty((1, 2))
    source_fraction[0, 0] = source.disk_fraction
    source_fraction[0, 1] = 1 - source.pointlike_fraction - source.disk_fraction
    source_hlr[0, 0] = source.disk_shape.half_light_radius.to(u.arcsec).value
    source_hlr[0, 1] = source.bulge_shape.half_light_radius.to(u.arcsec).value
    source_q[0, 0] = source.disk_shape.minor_major_axis_ratio
    source_q[0, 1] = source.bulge_shape.minor_major_axis_ratio
    source_beta[0, 0] = source.disk_shape.position_angle.to(u.deg).value
    source_beta[0, 1] = source.bulge_shape.position_angle.to(u.deg).value

    # Calculate fiberloss fractions.  Note that the calculator expects arrays
    # with implicit units.
    fiberloss_grid = calc.calculate(
        seeing_fwhm,
        scale.to(u.um / u.arcsec).value, offset.to(u.um).value,
        blur.to(u.um).value,
        source_fraction, source_hlr, source_q, source_beta,
        saved_images_file)

    if saved_table_file:
        meta = dict(
            description='Fiberloss fraction for source "{0}"'
            .format(source.name) +
            ' at focal (x,y) = ({0:.3f},{1:.3f})'
            .format(focal_x, focal_y))
        table = astropy.table.Table(meta=meta)
        table.add_column(astropy.table.Column(
            name='Wavelength', data=wlen_grid.value, unit=wlen_grid.unit,
            description='Observed wavelength'))
        table.add_column(astropy.table.Column(
            name='FiberAcceptance', data=fiberloss_grid[0],
            description='Fiber acceptance fraction'))
        args = {}
        if saved_table_file.endswith('.ecsv'):
            args['format'] = 'ascii.ecsv'
        table.write(saved_table_file, **args)

    # Interpolate (linearly) to the simulation wavelength grid.
    return np.interp(wavelength.data, wlen_grid.value, fiberloss_grid[0])
