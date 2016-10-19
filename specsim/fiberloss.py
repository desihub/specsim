# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate fiberloss fractions.

Fiberloss fractions are computed as the overlap between the light profile
illuminating a fiber and the on-sky aperture of the fiber.
"""
from __future__ import print_function, division

import astropy.units as u


def calculate_fiber_acceptance_fraction(source, atmosphere, instrument,
                                        observation):
    """
    """
    # Use tabulated when available.
    if instrument.fiber_acceptance_dict:
        return instrument.fiber_acceptance_dict[source.type_name]

    # Galsim is required to calculate fiberloss fractions on the fly.
    import galsim

    print('Will tabulate fiberloss at {0} wavelengths...'.format(
        instrument.fiberloss_ngrid))

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

    return instrument.fiber_acceptance_dict[source.type_name]
