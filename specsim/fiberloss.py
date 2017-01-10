# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate fiberloss fractions.

Fiberloss fractions are computed as the overlap between the light profile
illuminating a fiber and the on-sky aperture of the fiber.
"""
from __future__ import print_function, division

import numpy as np
import numpy.lib.stride_tricks

import astropy.units as u
import astropy.io.fits
import astropy.wcs


def calculate_fiber_acceptance_fraction(
    focal_x, focal_y, wavelength, source, atmosphere, instrument,
    oversampling = 16, save_images=None, save_table=None):
    """
    """
    # Use pre-tabulated fiberloss vs wavelength when available.
    if instrument.fiberloss_num_wlen == 0:
        return instrument.fiber_acceptance_dict[source.type_name]

    # Galsim is required to calculate fiberloss fractions on the fly.
    import galsim

    # Initialize the grid of wavelengths where the fiberloss will be
    # calculated.
    wlen_unit = wavelength.unit
    wlen_grid = np.linspace(wavelength.data[0], wavelength.data[-1],
                            instrument.fiberloss_num_wlen) * wlen_unit

    # Convert x, y offsets in length units to field angles.
    angle_x = (
        np.sign(focal_x) * instrument.field_radius_to_angle(np.abs(focal_x)))
    angle_y = (
        np.sign(focal_y) * instrument.field_radius_to_angle(np.abs(focal_y)))

    # Calculate radial offsets from the field center.
    focal_r = np.sqrt(focal_x ** 2 + focal_y ** 2)
    angle_r = np.sqrt(angle_x ** 2 + angle_y ** 2)

    # Calculate the plate scales in um/arcsec at this location.
    radial_scale = instrument.radial_scale(focal_r).to(u.um / u.arcsec).value
    azimuthal_scale = (instrument.azimuthal_scale(focal_r)
                       .to(u.um / u.arcsec).value)

    # Prepare an image of the fiber aperture for numerical integration.
    # Images are formed in the physical x-y space of the plate rather than
    # on sky angles.
    fiber_diameter = instrument.fiber_diameter.to(u.um).value
    num_pixels = instrument.fiberloss_num_pixels
    scale = fiber_diameter / num_pixels
    image = galsim.Image(num_pixels, num_pixels, scale=scale)

    # Create the instrument blur PSF and lookup the centroid offset at each
    # wavelength for this focal-plane position.
    blur_psf = []
    offsets = []
    for wlen in wlen_grid:
        # Lookup the RMS blur in focal-plane microns.
        blur_rms = instrument.get_blur_rms(wlen, angle_r).to(u.um).value
        # Use a Gaussian PSF to model blur.
        blur_psf.append(galsim.Gaussian(sigma=blur_rms))
        # Lookup the radial centroid offset in focal-plane microns.
        dx, dy = instrument.get_centroid_offset(angle_x, angle_y, wlen)
        offsets.append((dx.to(u.um).value, dy.to(u.um).value))

    # Create the atmospheric seeing model at each wavelength.
    seeing_psf = []
    for wlen in wlen_grid:
        # Lookup the seeing FWHM in arcsecs.
        seeing_fwhm = atmosphere.get_seeing_fwhm(wlen).to(u.arcsec).value
        # Use a Moffat profile to model the seeing in arcsec, then transform
        # to focal-plane microns.
        seeing_psf.append(galsim.Moffat(
            fwhm=seeing_fwhm, beta=atmosphere.seeing['moffat_beta']
            ).transform(radial_scale, 0, 0, azimuthal_scale).withFlux(1))

    # Create the source model, which we assume to be achromatic.
    source_components = []
    if source.pointlike_fraction > 0:
        source_components.append(galsim.Gaussian(
            flux=source.pointlike_fraction, sigma=1e-3 * scale))
    if source.disk_fraction > 0:
        hlr = source.disk_shape.half_light_radius.to(u.arcsec).value
        q = source.disk_shape.minor_major_axis_ratio
        beta = source.disk_shape.position_angle.to(u.deg).value
        source_components.append(galsim.Exponential(
            flux=source.disk_fraction, half_light_radius=hlr).shear(
                q=q, beta=beta * galsim.degrees))
    bulge_fraction = 1 - (source.pointlike_fraction + source.disk_fraction)
    if bulge_fraction > 0:
        hlr = source.bulge_shape.half_light_radius.to(u.arcsec).value
        q = source.bulge_shape.minor_major_axis_ratio
        beta = source.bulge_shape.position_angle.to(u.deg).value
        source_components.append(galsim.DeVaucouleurs(
            flux=bulge_fraction, half_light_radius=hlr).shear(
                q=q, beta=beta * galsim.degrees))
    # Combine the components and transform to focal-plane microns.
    gsparams = galsim.GSParams(maximum_fft_size=32767)
    source_model = galsim.Add(source_components, gsparams=gsparams).transform(
        radial_scale, 0, 0, azimuthal_scale).withFlux(1)

    # Prepare an anti-aliased image of the fiber aperture.
    nos = num_pixels * oversampling
    dxy = (np.arange(nos) + 0.5 - 0.5 * nos) / (0.5 * nos)
    rsq = dxy ** 2 + dxy[:, np.newaxis] ** 2
    inside = (rsq <= 1).astype(float)
    s0, s1 = inside.strides
    blocks = numpy.lib.stride_tricks.as_strided(
        inside, shape=(num_pixels, num_pixels, oversampling, oversampling),
        strides=(oversampling * s0, oversampling * s1, s0, s1))
    aperture = blocks.sum(axis=(2, 3)) / oversampling ** 2

    # Prepare to write a FITS file of images, if requested.
    if save_images:
        hdu_list = astropy.io.fits.HDUList()
        header = astropy.io.fits.Header()
        header['COMMENT'] = 'Fiberloss calculation images.'
        hdu_list.append(astropy.io.fits.PrimaryHDU(header=header))
        # All subsequent HDUs contain images with the same WCS.
        w = astropy.wcs.WCS(naxis=2)
        w.wcs.ctype = ['x', 'y']
        w.wcs.crpix = [num_pixels / 2. + 0.5, num_pixels / 2. + 0.5]
        w.wcs.cdelt = [scale, scale]
        w.wcs.crval = [0., 0.]
        header = w.to_header()

    # Build the convolved models and integrate. Save individual component
    # model images if requested.
    fiberloss_grid = np.empty(instrument.fiberloss_num_wlen)
    gsparams = galsim.GSParams(maximum_fft_size=32767)
    for i, wlen in enumerate(wlen_grid):
        convolved = galsim.Convolve([
            blur_psf[i], seeing_psf[i], source_model], gsparams=gsparams)
        # TODO: compare method='no_pixel' and 'auto' for accuracy and speed.
        dx, dy = offsets[i]
        offset = (dx / scale, dy / scale)
        draw_args = dict(image=image, method='auto')
        convolved.drawImage(offset=offset, **draw_args)
        fiberloss_grid[i] = np.sum(image.array * aperture)
        if save_images:
            header['WLEN'] = wlen.to(u.Angstrom).value
            header['FRAC'] = fiberloss_grid[i]
            header['COMMENT'] = 'Convolved model'
            hdu_list.append(astropy.io.fits.ImageHDU(
                data=image.array.copy(), header=header))
            # The component models are only rendered individually if we
            # need to save them.
            blur_psf[i].drawImage(offset=offset, **draw_args)
            header['COMMENT'] = 'Instrument blur model'
            hdu_list.append(astropy.io.fits.ImageHDU(
                data=image.array.copy(), header=header))
            # Render the seeing without the instrumental offset.
            seeing_psf[i].drawImage(**draw_args)
            header['COMMENT'] = 'Atmospheric seeing model'
            hdu_list.append(astropy.io.fits.ImageHDU(
                data=image.array.copy(), header=header))

    if save_images:
        # Render the source model without the instrumental offset.
        source_model.drawImage(**draw_args)
        header['COMMENT'] = 'Source model'
        hdu_list.append(astropy.io.fits.ImageHDU(
            data=image.array.copy(), header=header))
        header['COMMENT'] = 'Fiber aperture'
        hdu_list.append(astropy.io.fits.ImageHDU(
            data=aperture, header=header))
        hdu_list.writeto(save_images, clobber=True)

    if save_table:
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
            name='Fiberloss', data=fiberloss_grid,
            description='Fiber acceptance fraction'))
        args = {}
        if save_table.endswith('.ecsv'):
            args['format'] = 'ascii.ecsv'
        table.write(save_table, **args)

    return instrument.fiber_acceptance_dict[source.type_name]
