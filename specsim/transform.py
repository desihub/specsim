# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implement transformations between sky and focal plane coordinate systems.
"""
from __future__ import print_function, division

import numpy as np

import astropy.time
import astropy.coordinates
import astropy.constants
from astropy import units as u


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
        alt(float or numpy.ndarray): Target altitude(s) in radians above the
            horizon.
        az(float numpy.ndarray): Target azimuthal angle(s) in radians east of
            north.
        alt0(float or numpy.ndarray): Boresight altitude(s) in radians above
            the horizon.
        az0(numpy.ndarray): Boresight azimuthal angle(s) in radians east of
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
    ##u = (np.sin(az) * cos_alt, np.cos(az) * cos_alt, np.sin(alt))

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

    # Convert unit vectors to (x,y).
    return v[0], v[2]
