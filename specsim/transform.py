# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implement transformations between sky and focal plane coordinate systems.

Attributes:
    observatories (dict): Dictionary of predefined observing locations
        represented as :class:`astropy.coordinates.EarthLocation` objects.
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


def altaz_to_focalplane(alt, az, alt0, az0):
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

    All input values are assumed to be in radians and all output values are
    calculated in radians.

    Args:
        alt (float or numpy.ndarray): Target altitude(s) in radians above the
            horizon.
        az (float numpy.ndarray): Target azimuthal angle(s) in radians east of
            north.
        alt0 (float or numpy.ndarray): Boresight altitude(s) in radians above
            the horizon.
        az0 (numpy.ndarray): Boresight azimuthal angle(s) in radians east of
            north.

    Returns:
        tuple: Pair x,y of numpy arrays of focal-plane coordinates in radians,
            with +x along the azimuth direction (increasing eastwards) and +y
            along the altitude direction (increasing towards zenith). The
            output arrays have the same shapes, given by
            :func:`np.broadcast(alt, az, alt0, az0) <numpy.broadcast>`.
    """
    if not isinstance(alt, np.ndarray):
        alt = np.float(alt)
    if not isinstance(az, np.ndarray):
        az = np.float(az)
    if not isinstance(alt0, np.ndarray):
        alt0 = np.float(alt0)
    if not isinstance(az0, np.ndarray):
        az0 = np.float(az0)

    # Check that the input shapes are compatible for broadcasting to the output,
    # otherwise this will raise a ValueError.
    output_shape = np.broadcast(alt, az, alt0, az0).shape

    # Convert (alt,az) to unit vectors.
    cos_alt = np.cos(alt)
    elem_shape = np.broadcast(alt, az).shape
    u = np.empty(shape=[3,] + list(elem_shape))
    u[0] = np.sin(az) * cos_alt
    u[1] = np.cos(az) * cos_alt
    u[2] = np.sin(alt)

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

    # Calculate v = R.u
    v = np.einsum('ij...,j...->i...', R, u)
    if v[0].shape != output_shape:
        raise RuntimeError(
            'np.einsum does not broadcast correctly in numpy {}.'
            .format(np.version.version))

    # Convert unit vectors to (x,y).
    return v[0], v[2]


def focalplane_to_altaz(x, y, alt0, az0):
    """Convert focal plane (x,y) coordinates to local (alt,az) coordinates.

    This is the inverse of :func:`altaz_to_focalplane`. Consult that function's
    documentation for details.

    Args:
        x (float or numpy.ndarray): Target x position(s) in radians with +x
            increasing eastwards along the azimuth direction.
        y (float numpy.ndarray): Target y position(s) in radians with +y
            increasing towards the zenith along the altitude direction.
        alt0 (float or numpy.ndarray): Boresight altitude(s) in radians above
            the horizon.
        az0 (numpy.ndarray): Boresight azimuthal angle(s) in radians east of
            north.

    Returns:
        tuple: Pair alt,az of numpy arrays of local sky coordinates in radians,
            with alt measured above the horizon and az increasing eastwards of
            north. The output arrays have the same shapes, given by
            :func:`np.broadcast(x, y, alt0, az0) <numpy.broadcast>`.
    """
    if not isinstance(x, np.ndarray):
        x = np.float(x)
    if not isinstance(y, np.ndarray):
        y = np.float(y)
    if not isinstance(alt0, np.ndarray):
        alt0 = np.float(alt0)
    if not isinstance(az0, np.ndarray):
        az0 = np.float(az0)

    # Convert (x,y) to unit vectors.
    z = np.sqrt(1 - x**2 - y**2)
    v = np.empty(shape=[3,] + list(z.shape))
    v[0] = x
    v[1] = z
    v[2] = y

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

    # Calculate u = R.v
    u = np.einsum('ij...,j...->i...', R, v)

    # Convert unit vectors to (alt,az).
    alt = np.arcsin(u[2])
    az = np.arctan2(u[0], u[1])
    return alt, az


def sky_to_altaz(sky_coords, where, when, wavelength, temperature=15*u.deg_C,
                 pressure=None, relative_humidity=0):
    """Convert sky coordinates to (alt,az) for specified observing conditions.

    This function encapsulates algorithms for the time-dependent transformation
    between RA-DEC and ALT-AZ, and models the wavelength-dependent atmospheric
    refraction.

    The output shape is determined by the usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__ applied
    to all of the inputs.

    Args:
        sky_coords: An object representing one or more sky coordinates that are
            transformable to an AltAz frame by invoking
            ``sky_coords.transform_to()``.  This argument will usually be an
            instances of :class:`astropy.coordinates.SkyCoord`, but instances
            of :class:`astropy.coordinates.AltAz` can also be used to isolate
            the effects of changing the parameters of the atmospheric
            refraction model.
        where (astropy.coordinates.EarthLocation): The location where the
            observations take place.
        when (astropy.time.Time): The time(s) of the observations.
        wavelength (astropy.units.Quantity): The wavelength(s) of the
            observations.

        temperature (astropy.units.Quantity): The temperature(s) of the
            observations.
        pressure (astropy.units.Quantity): The atmospheric pressure(s) of the
            observations. These should be pressures at the telescope, rather
            than adjusted to equivalent sea-level pressures. When ``None`` is
            specified, the pressure(s) will be estimated at the telescope
            elevation using a standard atmosphere model at the specified
            temperature(s).
        relative_humidity( float or numpy.ndarray): Relative humidity (or
            humidities) of the observations. Value(s) should be in the range
            0-1 and are dimensionless.

    Returns:
        astropy.coordinates.AltAz: An array of ALT-AZ coordinates with a shape
            given by
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
