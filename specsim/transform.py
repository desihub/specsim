# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implement transformations between sky and focal plane coordinate systems.

Attributes
----------
observatories : dict
    Dictionary of predefined observing locations represented as
    :class:`astropy.coordinates.EarthLocation` objects.
"""
from __future__ import print_function, division

import numpy as np

import astropy.time
import astropy.coordinates
import astropy.constants
from astropy import units as u


observatories = {
    'APO': astropy.coordinates.EarthLocation.from_geodetic(
        lat='32d46m49s', lon='-105d49m13s', height=2788.*u.m),
    'KPNO': astropy.coordinates.EarthLocation.from_geodetic(
        lat='31d57m48s', lon='-111d36m0s', height=2120.*u.m),
}


def altaz_to_focalplane(alt, az, alt0, az0, platescale=1):
    """
    Convert local (alt,az) coordinates to focal plane (x,y) coordinates.

    A plate coordinate system is defined by its boresight altitude and azimuth,
    corresponding to (x,y) = (0,0), and the conventions that +x increases
    eastwards along the azimuth axis and +y increases towards the zenith along
    the altitude axis.

    This function implements a purely mathematical coordinate transform and does
    not invoke any atmospheric refraction physics.  Use :func:`sky_to_altaz`
    to convert global sky coordinates (ra,dec) into local (alt,az) coordinates,
    which does involve refraction.

    The input values can either be constants or numpy arrays. If any input is
    not a numpy array, it will be automatically converted to a
    :class:`numpy.float`. The output shape is determined by the usual
    `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__
    applied to all of the inputs.

    Parameters
    ----------
    alt : :class:`astropy.coordinates.Angle`
        Target altitude angle(s) above the horizon.
    az : :class:`astropy.coordinates.Angle`
        Target azimuthal angle(s) east of north.
    alt0 : :class:`astropy.coordinates.Angle`
        Boresight altitude angle(s) above the horizon.
    az0 : :class:`astropy.coordinates.Angle`
        Boresight azimuthal angle(s) east of north.
    platescale : :class:`astropy.units.Quantity`
        Conversion from angular separation relative to the boresight to
        the output focal plane coordinates.

    Returns
    -------
    :class:`tuple`
        Pair x,y of focal-plane coordinates expressed as
        :class:`astropy.units.Quantity` objects, with +x along the
        azimuth direction (increasing eastwards) and +y along the altitude
        direction (increasing towards zenith). The output arrays have the same
        shapes, given by
        :func:`np.broadcast(alt, az, alt0, az0) <numpy.broadcast>`. The output
        units are determined by the input ``platescale`` and will be ``u.rad``
        if the platescale is dimensionless, or otherwise the SI units of
        ``platescale * u.rad``.
    """
    # Check that the input shapes are compatible for broadcasting to the output,
    # otherwise this will raise a ValueError.
    output_shape = np.broadcast(alt, az, alt0, az0).shape

    # Convert (alt,az) to unit vectors.
    cos_alt = np.cos(alt)
    elem_shape = np.broadcast(alt, az).shape
    uu = np.empty(shape=[3,] + list(elem_shape))
    uu[0] = np.sin(az) * cos_alt
    uu[1] = np.cos(az) * cos_alt
    uu[2] = np.sin(alt)

    # Build combined rotation matrices R[-alt0,x].R[+az0,z].
    cos_alt0 = np.cos(alt0)
    sin_alt0 = np.sin(alt0)
    cos_az0 = np.cos(az0)
    sin_az0 = np.sin(az0)
    elem_shape = np.broadcast(alt0, az0).shape
    R = np.empty(shape=[3,3] + list(elem_shape))
    R[0, 0] = cos_az0
    R[0, 1] = -sin_az0
    R[0, 2] = 0.
    R[1, 0] = cos_alt0 * sin_az0
    R[1, 1] = cos_alt0 * cos_az0
    R[1, 2] = sin_alt0
    R[2, 0] = -sin_alt0 * sin_az0
    R[2, 1] = -cos_az0 * sin_alt0
    R[2, 2] = cos_alt0

    # Calculate vv = R.uu
    vv = np.einsum('ij...,j...->i...', R, uu)
    if vv[0].shape != output_shape:
        raise RuntimeError(
            'np.einsum does not broadcast correctly in numpy {}.'
            .format(np.version.version))

    # Convert unit vectors to (x,y).
    conversion = (1 * u.rad * platescale).si
    return vv[0] * conversion, vv[2] * conversion


def focalplane_to_altaz(x, y, alt0, az0, platescale=1):
    """Convert focal plane (x,y) coordinates to local (alt,az) coordinates.

    This is the inverse of :func:`altaz_to_focalplane`. Consult that function's
    documentation for details.

    Parameters
    ----------
    x : :class:`astropy.units.Quantity`
        Target x position(s) in the focal plane with +x increasing eastwards
        along the azimuth direction.  The input units must be such that
        ``x / platescale`` is an angle.
    y : :class:`astropy.units.Quantity`
        Target y position(s) in focal plane with +y increasing towards the
        zenith along the altitude direction.  The input units must be such that
        ``y / platescale`` is an angle.
    alt0 : :class:`astropy.coordinates.Angle`
        Boresight altitude angle(s) above the horizon.
    az0 : :class:`astropy.coordinates.Angle`
        Boresight azimuthal angle(s) east of north.
    platescale : :class:`astropy.units.Quantity`
        Conversion from angular separation relative to the boresight to
        the output focal plane coordinates.

    Returns
    -------
    :class:`tuple`
        Pair alt,az of focal-plane coordinates expressed as
        :class:`astropy.units.Angle` objects, with alt measured above the
        horizon and az increasing eastwards of north. The output arrays have
        the same shapes, given by
        :func:`np.broadcast(x, y, alt0, az0) <numpy.broadcast>`.
    """
    # Convert (x,y) to vectors in radians.
    x = (x / platescale).to(u.rad)
    y = (y / platescale).to(u.rad)
    z = np.sqrt(1 - x.value**2 - y.value**2)
    vv = np.empty(shape=[3,] + list(z.shape))
    vv[0] = x.value
    vv[1] = z
    vv[2] = y.value

    # Build combined rotation matrices R[-alt0,x].R[+az0,z].
    cos_alt0 = np.cos(alt0)
    sin_alt0 = np.sin(alt0)
    cos_az0 = np.cos(az0)
    sin_az0 = np.sin(az0)
    elem_shape = np.broadcast(alt0, az0).shape
    R = np.empty(shape=[3,3] + list(elem_shape))
    R[0, 0] = cos_az0
    R[0, 1] = cos_alt0 * sin_az0
    R[0, 2] = -sin_alt0 * sin_az0
    R[1, 0] = -sin_az0
    R[1, 1] = cos_alt0 * cos_az0
    R[1, 2] = -cos_az0 * sin_alt0
    R[2, 0] = 0.
    R[2, 1] = sin_alt0
    R[2, 2] = cos_alt0

    # Calculate uu = R.vv
    uu = np.einsum('ij...,j...->i...', R, vv)

    # Convert unit vectors to (alt,az).
    alt = np.arcsin(uu[2])
    az = np.arctan2(uu[0], uu[1])
    return alt * u.rad, az * u.rad


def sky_to_altaz(sky_coords, where, when, wavelength, temperature=15*u.deg_C,
                 pressure=None, relative_humidity=0):
    """Convert sky coordinates to (alt,az) for specified observing conditions.

    This function encapsulates algorithms for the time-dependent transformation
    between RA-DEC and ALT-AZ, and models the wavelength-dependent atmospheric
    refraction.

    The output shape is determined by the usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__ applied
    to all of the inputs.

    Parameters
    ----------
    sky_coords : :class:`object`
        An object representing one or more sky coordinates that are
        transformable to an AltAz frame by invoking
        ``sky_coords.transform_to()``. This argument will usually be an
        instances of :class:`astropy.coordinates.SkyCoord`, but instances
        of :class:`astropy.coordinates.AltAz` can also be used to isolate
        the effects of changing the parameters of the atmospheric
        refraction model.
    where : :class:`astropy.coordinates.EarthLocation`
        The location where the observations take place.
    when : :class:`astropy.time.Time`
        The time(s) of the observations.
    wavelength : :class:`astropy.units.Quantity`
        The wavelength(s) of the observations with units of length.
    temperature : :class:`astropy.units.Quantity`
        The temperature(s) of the observations with temperature units.
    pressure : :class:`astropy.units.Quantity`
        The atmospheric pressure(s) of the observations with appropriate units.
        These should be pressures at the telescope, rather
        than adjusted to equivalent sea-level pressures. When ``None`` is
        specified, the pressure(s) will be estimated at the telescope elevation
        using a standard atmosphere model at the specified temperature(s).
    relative_humidity : :class:`float` or :class:`numpy.ndarray`
        Relative humidity (or humidities) of the observations. Value(s) should
        be in the range 0-1 and are dimensionless.

    Returns
    -------
    :class:`astropy.coordinates.AltAz`
        An array of ALT-AZ coordinates with a shape given by
        :func:`np.broadcast(sky_coords, when, wavelength, temperature, pressure)
        <numpy.broadcast>`.
    """
    if not isinstance(relative_humidity, np.ndarray):
        relative_humidity = np.float(relative_humidity)
    if np.any((relative_humidity < 0) | (relative_humidity > 1)):
        raise ValueError('Values of relative_humidity must be 0-1.')

    # Convert temperature(s).
    T_in_C = temperature.to(u.deg_C, equivalencies=u.temperature())

    # Estimate pressure(s) based on elevation, if necessary.
    # See https://en.wikipedia.org/wiki/Vertical_pressure_variation
    if pressure is None:
        h = where.height
        p0 = astropy.constants.atmosphere
        g0 = astropy.constants.g0
        R = astropy.constants.R
        air_molar_mass = 0.0289644 * u.kg / u.mol
        T_in_K = temperature.to(u.K, equivalencies=u.temperature())
        pressure = p0 * np.exp(-h * air_molar_mass * g0 / (R * T_in_K))

    # Check that the input shapes are compatible for broadcasting to the output,
    # otherwise this will raise a ValueError.
    output_shape = np.broadcast(sky_coords, when, wavelength,
                                temperature, pressure).shape

    # Initialize the altaz frames for each (time, wavelength, temperature,
    # pressure, relative_humidity).
    observing_frame = astropy.coordinates.AltAz(
        location=where, obstime=when, obswl=wavelength, temperature=T_in_C,
        pressure=pressure, relative_humidity=relative_humidity)

    # Perform the transforms.
    return sky_coords.transform_to(observing_frame)


def adjust_time_to_hour_angle(nominal_time, target_ra, hour_angle,
                              longitude=None,
                              max_error=0.01*u.arcsec, max_iterations=3):
    """Adjust a time to a specified target hour angle.

    The input nominal time will be adjusted to the closest time where the
    specified hour angle is achieved, using either a positive or negative
    adjustment.

    Parameters
    ----------
    nominal_time : :class:`astropy.time.Time`
        Nominal time that will be adjusted. If it does not have an associated
        location, the longitude parameter must be set.
    target_ra : :class:`astropy.units.quantity.Quantity`
        Target right ascension to use for calculating the hour angle.
    hour_angle : :class:`astropy.units.quantity.Quantity`
        Desired target hour angle after the adjustment, expressed as an angle.
    longitude : :class:`astropy.units.quantity.Quantity`
        The longitude to use for calculating the hour angle.  When the value
        is ``None``, the location associated with ``nominal_time`` is used.
    max_error : :class:`astropy.units.quantity.Quantity`
        The desired accuracy of the hour angle after the adjustment, expressed
        as an angle.
    max_iterations : int
        The maximum number of iterations to use in order to achieve the desired
        accuracy.

    Returns
    -------
    :class:`astropy.time.Time`
        Adjusted time, which will be within ``max_error`` of ``target_ra``, or
        else a RuntimeError will be raised.

    Raises
    ------
    :class:`RuntimeError`
        The desired accuracy could not be achieved after ``max_iterations``.
    """
    sidereal = 1 / 1.002737909350795
    when = nominal_time
    num_iterations = 0
    while True:
        # Calculate the nominal local sidereal time of the target.
        lst = when.sidereal_time('apparent', longitude) - target_ra

        # Are we close enough?
        if np.abs(lst - hour_angle) <= max_error:
            break

        # Have we run out of iterations?
        if num_iterations >= max_iterations:
            raise RuntimeError(
                'Reached max_iterations = {}.'.format(max_iterations))
        num_iterations += 1

        # Offset to the nearest time with the desired hour angle.
        # Correct for the fact that 360 deg corresponds to a sidereal day.
        when = when - (lst - hour_angle) * u.hour / (15 * u.deg) * sidereal

    return when
