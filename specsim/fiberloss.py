# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate fiberloss fractions.

Fiberloss fractions are computed as the overlap between the light profile
illuminating a fiber and the on-sky aperture of the fiber.
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u
import astropy.io.fits
import astropy.wcs


def calculate_fiber_acceptance_fraction(
    focal_x, focal_y, wavelength, source, atmosphere, instrument, save=None):
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

    # Calculate the field angle from the focal-plane (x,y).
    focal_r = np.sqrt(focal_x ** 2 + focal_y ** 2)
    angle = instrument.field_radius_to_angle(focal_r)

    # Calculate the plate scales in um/arcsec at this location.
    radial_scale = instrument.radial_scale(focal_r).to(u.um / u.arcsec).value
    azimuthal_scale = (instrument.azimuthal_scale(focal_r)
                       .to(u.um / u.arcsec).value)

    # Prepare an image of the fiber aperture for numerical integration.
    # Images are formed in the physical x-y space of the plate rather than
    # on sky angles.
    fiber_diameter = instrument.fiber_diameter.to(u.um).value
    scale = 1.05 * fiber_diameter / 100
    npix_r = 100
    npix_phi = 100
    image = galsim.Image(npix_r, npix_phi, scale=scale)

    # Create the instrument blur PSF and lookup the centroid offset at each
    # wavelength for this focal-plane position.
    blur_psf = []
    offsets = []
    for wlen in wlen_grid:
        # Lookup the RMS blur in focal-plane microns.
        blur_rms = instrument.get_blur_rms(wlen, angle).to(u.um).value
        # Use a Gaussian PSF to model blur.
        blur_psf.append(galsim.Gaussian(sigma=blur_rms))
        # Lookup the radial centroid offset in focal-plane microns.
        offsets.append(
            instrument.get_centroid_offset(wlen, angle).to(u.um).value)

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

    # Calculate the coordinates at the center of each image pixel relative to
    # the fiber center.
    dr = (np.arange(npix_r) - 0.5 * npix_r - 0.5) * scale
    dphi = (np.arange(npix_phi) - 0.5 * npix_phi - 0.5) * scale

    # Select pixels whose center is within the fiber aperture.
    rsq = dr ** 2 + dphi[:, np.newaxis] ** 2
    inside = (rsq < (fiber_diameter / 2) ** 2)

    # Prepare to write a FITS file of images, if requested.
    if save:
        hdu_list = astropy.io.fits.HDUList()
        header = astropy.io.fits.Header()
        header['COMMENT'] = 'Fiberloss calculation images.'
        hdu_list.append(astropy.io.fits.PrimaryHDU(header=header))
        # All subsequent HDUs contain images with the same WCS.
        w = astropy.wcs.WCS(naxis=2)
        w.wcs.ctype = ['x', 'y']
        w.wcs.crpix = [npix_r / 2. + 0.5, npix_phi / 2. + 0.5]
        w.wcs.cdelt = [scale, scale]
        w.wcs.crval = [0., 0.]
        header = w.to_header()

    # Build the convolved models and integrate. Save individual component
    # models if requested.
    gsparams = galsim.GSParams(maximum_fft_size=32767)
    for i, wlen in enumerate(wlen_grid):
        convolved = galsim.Convolve([
            blur_psf[i], seeing_psf[i], source_model], gsparams=gsparams)
        # TODO: compare method='no_pixel' and 'auto' for accuracy and speed.
        draw_args = dict(
            image=image, method='auto', offset=(offsets[i], 0.))
        convolved.drawImage(**draw_args)
        fraction = np.sum(image.array * inside)
        print('fiberloss:', wlen, offsets[i], fraction)
        if save:
            header['WLEN'] = wlen.to(u.Angstrom).value
            header['FRAC'] = fraction
            header['COMMENT'] = 'Convolved model'
            hdu_list.append(astropy.io.fits.ImageHDU(
                data=image.array.copy(), header=header))
            # The component models are only rendered individually if we
            # need to save them.
            blur_psf[i].drawImage(**draw_args)
            header['COMMENT'] = 'Instrument blur model'
            hdu_list.append(astropy.io.fits.ImageHDU(
                data=image.array.copy(), header=header))
            seeing_psf[i].drawImage(**draw_args)
            header['COMMENT'] = 'Atmospheric seeing model'
            hdu_list.append(astropy.io.fits.ImageHDU(
                data=image.array.copy(), header=header))

    if save:
        del draw_args['offset']
        source_model.drawImage(**draw_args)
        header['COMMENT'] = 'Source model'
        hdu_list.append(astropy.io.fits.ImageHDU(
            data=image.array.copy(), header=header))
        image.array[:] = inside
        header['COMMENT'] = 'Fiber aperture'
        hdu_list.append(astropy.io.fits.ImageHDU(
            data=image.array.copy(), header=header))
        hdu_list.writeto(save, clobber=True)

    return instrument.fiber_acceptance_dict[source.type_name]
