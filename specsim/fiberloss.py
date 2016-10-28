# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate fiberloss fractions.

Fiberloss fractions are computed as the overlap between the light profile
illuminating a fiber and the on-sky aperture of the fiber.
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u


def calculate_fiber_acceptance_fraction(
    focal_x, focal_y, wavelength, source, atmosphere, instrument, save=None):
    """
    """
    # Use pre-tabulated fiberloss vs wavelength when available.
    if instrument.fiberloss_ngrid == 0:
        return instrument.fiber_acceptance_dict[source.type_name]

    # Galsim is required to calculate fiberloss fractions on the fly.
    import galsim

    wlen_unit = wavelength.unit
    wlen_grid = np.linspace(wavelength.data[0], wavelength.data[-1],
                            instrument.fiberloss_ngrid) * wlen_unit

    # Calculate the field angle from the focal-plane (x,y).
    focal_r = np.sqrt(focal_x ** 2 + focal_y ** 2)
    angle = instrument.field_radius_to_angle(focal_r)

    # Create the instrument blur PSF and lookup the centroid offset at each
    # wavelength for this focal-plane position.
    blur_psf = []
    offsets = []
    for wlen in wlen_grid:
        blur_rms = instrument.get_blur_rms(wlen, angle)
        # Convert to an angular size on the sky ignoring any asymmetry that
        # might be introduced by different radial and azimuthal plate scales.
        rscale = instrument.radial_scale(focal_r)
        blur_rms /= rscale
        blur_psf.append(galsim.Gaussian(sigma=blur_rms.to(u.arcsec).value))
        offset = instrument.get_centroid_offset(wlen, angle)
        # Convert to an angular offset on the sky.
        offset /= rscale
        offsets.append(offset.to(u.arcsec).value)

    # Create the atmospheric seeing model at each wavelength.
    seeing_psf = []
    for wlen in wlen_grid:
        seeing_psf.append(galsim.Moffat(
            fwhm=atmosphere.get_seeing_fwhm(wlen).to(u.arcsec).value,
            beta=atmosphere.seeing['moffat_beta']))

    # Create the source model, which we assume to be achromatic.
    source_components = []
    if source.disk_fraction > 0:
        hlr = source.disk_shape.half_light_radius.to(u.arcsec).value
        q = source.disk_shape.minor_major_axis_ratio
        beta = source.disk_shape.position_angle.to(u.deg).value
        disk_model = galsim.Exponential(
            flux=source.disk_fraction, half_light_radius=hlr).shear(
                q=q, beta=beta * galsim.degrees)
    if source.disk_fraction < 1:
        hlr = source.bulge_shape.half_light_radius.to(u.arcsec).value
        q = source.bulge_shape.minor_major_axis_ratio
        beta = source.bulge_shape.position_angle.to(u.deg).value
        bulge_model = galsim.DeVaucouleurs(
            flux=source.disk_fraction, half_light_radius=hlr).shear(
                q=q, beta=beta * galsim.degrees)
    if source.disk_fraction == 0:
        source_model = bulge_model
    elif source.disk_fraction == 1:
        source_model = disk_model
    else:
        source_model = disk_model + bulge_model

    # Calculate the on-sky fiber aperture.
    radial_size = (0.5 * instrument.fiber_diameter /
                   instrument.radial_scale(focal_r)).to(u.arcsec).value
    azimuthal_size = (0.5 * instrument.fiber_diameter /
                      instrument.azimuthal_scale(focal_r)).to(u.arcsec).value

    # Build the convolved model.
    gsparams = galsim.GSParams(maximum_fft_size=32767)
    convolved = []
    for i, wlen in enumerate(wlen_grid):
        convolved.append(galsim.Convolve([
            blur_psf[i], seeing_psf[i], source_model], gsparams=gsparams))

    return instrument.fiber_acceptance_dict[source.type_name]
