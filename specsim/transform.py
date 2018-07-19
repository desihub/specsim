# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implement transformations between sky and focal plane coordinate systems.

The :func:`sky_to_altaz` and :func:`altaz_to_sky` functions use the
atmospheric refraction model implemented in :class:`astropy.coordinates.AltAz`,
based on that implemented in `ERFA <https://github.com/liberfa/erfa>`__, which
is fast but becomes inaccurate at low altitudes (high airmass).  The table
below (from `this notebook <https://github.com/desihub/specsim/
blob/master/docs/nb/TransformExamples.ipynb>`__) gives the round-trip altitude
errors for converting from (alt,az) to (ra,dec) and back to (alt,az):

============== ==============
Altitude (deg) Error (arcsec)
============== ==============
          10.0        -15.175
          15.0         -1.891
          20.0         -0.425
          25.0         -0.130
          30.0         -0.048
          35.0         -0.020
          40.0         -0.009
          45.0         -0.004
============== ==============

The :func:`sky_to_altaz` and :func:`altaz_to_sky` issue a warning when using
altitude angles (and a non-zero atmospheric pressure) below 20 degrees.  The
value of this threshold can be read and changed programmatically via the
:attr:`low_altitude_threshold` module attribute.

Attributes
----------
observatories : dict
    Dictionary of predefined observing locations represented as
    :class:`astropy.coordinates.EarthLocation` objects.
low_altitude_threshold : :class:`astropy.coordinates.Angle`
    The atmospheric refraction model becomes inaccurate for altitude angles
    below this threshold, so :func:`sky_to_altaz` and :func:`altaz_to_sky`
    will issue a `UserWarning` if they encounter a lower value, unless
    refraction has been disabled by specifying zero pressure.
"""
from __future__ import print_function, division

import warnings

import numpy as np

import astropy.time
import astropy.coordinates
import astropy.constants
from astropy import units as u

try:
    basestring          #- exists in py2
except NameError:
    basestring = str    #- for py3

observatories = {
    'APO': astropy.coordinates.EarthLocation.from_geodetic(
        lat='32d46m49s', lon='-105d49m13s', height=2788.*u.m),
    'KPNO': astropy.coordinates.EarthLocation.from_geodetic(
        lat='31d57m48s', lon='-111d36m0s', height=2120.*u.m),
    # http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon
    'LSST': astropy.coordinates.EarthLocation.from_geodetic(
        lat='-30d14m40.68s', lon='-70d44m57.90s', height=2647.*u.m),
}


low_altitude_threshold = 5 * u.deg


def altaz_to_focalplane(alt, az, alt0, az0, platescale=1):
    """
    Convert local (alt,az) coordinates to focal plane (x,y) coordinates.

    A plate coordinate system is defined by its boresight altitude and azimuth,
    corresponding to the (x,y) origin, and the conventions that +x increases
    eastwards along the azimuth axis and +y increases towards the zenith along
    the altitude axis:

    >>> scale = 200 * u.mm / u.deg
    >>> alt0, az0 = 45 * u.deg, 0 * u.deg
    >>> x, y = altaz_to_focalplane(alt0 + 1 * u.deg, az0, alt0, az0, scale)
    >>> print('X = %.2f mm, Y = %.2f mm' % (x.to(u.mm).value, y.to(u.mm).value))
    X = 0.00 mm, Y = 199.99 mm
    >>> x, y = altaz_to_focalplane(alt0, az0 + 1 * u.deg, alt0, az0, scale)
    >>> print('X = %.2f mm, Y = %.2f mm' % (x.to(u.mm).value, y.to(u.mm).value))
    X = 141.41 mm, Y = 0.87 mm

    This function implements a purely mathematical coordinate transform and does
    not invoke any atmospheric refraction physics.  Use :func:`sky_to_altaz`
    to convert global sky coordinates (ra,dec) into local (alt,az) coordinates,
    which does involve refraction.

    The output shape is determined by the usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__
    applied to all of the inputs, , so this function can be used to tabulate
    (x,y) coordinates on a user-specified grid covering different targets
    and boresights.

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

    This is the inverse of :func:`altaz_to_focalplane`:

    >>> scale = 200 * u.mm / u.deg
    >>> alt0, az0 = 45 * u.deg, 0 * u.deg
    >>> x, y = 4 * u.mm, -2 * u.mm
    >>> alt, az = focalplane_to_altaz(x, y, alt0, az0, scale)
    >>> x, y = altaz_to_focalplane(alt, az, alt0, az0, scale)
    >>> print('X = %.2f mm, Y = %.2f mm' % (x.to(u.mm).value, y.to(u.mm).value))
    X = 4.00 mm, Y = -2.00 mm

    Consult that function's documentation for details.

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


def create_observing_model(where, when, wavelength, temperature=15*u.deg_C,
                           pressure=None, relative_humidity=0):
    """Create a model for observations through the atmosphere.

    This function encapsulates algorithms for the time-dependent transformation
    between sky coordinates and ALT-AZ through a specified atmosphere, and
    models the wavelength-dependent atmospheric refraction.

    The model returned by this function can be passed to :func:`sky_to_altaz`
    and :func:`altaz_to_sky` to transform sky coordinates such as RA,DEC to
    and from this model's ALT,AZ coordinate frame(s).

    The output shape resulting from using a model with :func:`sky_to_altaz` or
    :func:`altaz_to_sky` is determined by the usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__ applied
    to all of the coordinates and observing model inputs, so this function can
    be used to tabulate an observing model on a user-specified grid covering
    location x time x wavelength x temperature x pressure x humidity.  For
    example, to model a grid of observing times and wavelengths:

    >>> where = observatories['KPNO']
    >>> when = astropy.time.Time([56382.9, 56383.1], format='mjd')
    >>> wlen = np.linspace(4000., 10000., 7) * u.Angstrom
    >>> obs_model = create_observing_model(where, when[:, np.newaxis], wlen)
    >>> sky_in = astropy.coordinates.ICRS(ra=45*u.deg, dec=45*u.deg)
    >>> sky_to_altaz(sky_in, obs_model).shape
    (2, 7)

    Parameters
    ----------
    where : :class:`astropy.coordinates.EarthLocation`
        The location(s) on the earth where the sky is being observed.
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
        An array of ALT-AZ coordinate frames with a shape given by
        :func:`np.broadcast(where, when, wavelength, temperature, pressure,
        relative_humidity) <numpy.broadcast>`.
    """
    relative_humidity = np.asarray(relative_humidity)
    if np.any((relative_humidity < 0) | (relative_humidity > 1)):
        raise ValueError('Values of relative_humidity must be 0-1.')

    # Convert temperature(s).
    T_in_C = temperature.to(u.deg_C, equivalencies=u.temperature())

    # Estimate pressure(s) based on elevation, if necessary.
    # See https://en.wikipedia.org/wiki/Vertical_pressure_variation
    if pressure is None:
        h = where.height
        # The atmosphere constant in astropy < 2.0 was renamed to atm in 2.0.
        try:
            p0 = astropy.constants.atm
        except AttributeError:
            # Fallback for astropy < 2.0.
            p0 = astropy.constants.atmosphere
        g0 = astropy.constants.g0
        R = astropy.constants.R
        air_molar_mass = 0.0289644 * u.kg / u.mol
        T_in_K = temperature.to(u.K, equivalencies=u.temperature())
        pressure = p0 * np.exp(-h * air_molar_mass * g0 / (R * T_in_K))

    # Check that the input shapes are compatible for broadcasting to the output,
    # otherwise this will raise a ValueError.
    np.broadcast(where, when, wavelength, temperature, pressure,
                 relative_humidity)

    # Initialize the altaz frames for each (time, wavelength, temperature,
    # pressure, relative_humidity).
    return astropy.coordinates.AltAz(
        location=where, obstime=when, obswl=wavelength, temperature=T_in_C,
        pressure=pressure, relative_humidity=relative_humidity)


def sky_to_altaz(sky_coords, observing_model):
    """Convert sky coordinates to (alt,az) for specified observing conditions.

    Transformations between sky coordinates such as RA,DEC and ALT,AZ involve
    two steps.  First, define the atmosphere through which the sky is being
    observed with a call to :func:`create_observing_model`.  Next, call this
    function. For example, to locate Polaris from Kitt Peak, observing at
    a wavelength of 5400 Angstroms (we expect altitude ~ longitude ~ 32 deg):

    >>> where = observatories['KPNO']
    >>> when = astropy.time.Time('2001-01-01T00:00:00')
    >>> obs_model = create_observing_model(where, when, 5400 * u.Angstrom)
    >>> polaris = astropy.coordinates.ICRS(ra=37.95 * u.deg, dec=89.25 * u.deg)
    >>> altaz = sky_to_altaz(polaris, obs_model)
    >>> print('alt = %.3f deg, az = %.3f deg' %
    ... (altaz.alt.to(u.deg).value, altaz.az.to(u.deg).value))
    alt = 32.465 deg, az = 0.667 deg

    The output shape is determined by the usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__ applied
    to the input coordinates and the observing model input parameters.

    Setting a pressure value of zero disables the atmospheric refraction model,
    so that returned coordinates are topocentric.  The atmospheric refraction
    model becomes inaccurate for altitudes below 20 degrees so a `UserWarning`
    will be issued to flag this condition.

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
    observing_model : :class:`astropy.coordinates.AltAz`
        The ALT-AZ coordinate frame(s) that specify the observing conditions,
        normally obtained by calling :func:`create_observing_model`.

    Returns
    -------
    :class:`astropy.coordinates.AltAz`
        An array of ALT-AZ coordinates with a shape given by
        :func:`np.broadcast(sky_coords, observing_model) <numpy.broadcast>`.
    """
    # Check that the input coordinates are compatible for broadcasting with
    # the refraction model, otherwise this will raise a ValueError.
    output_shape = np.broadcast(
        sky_coords, observing_model.location, observing_model.obstime,
        observing_model.obswl, observing_model.temperature,
        observing_model.pressure, observing_model.relative_humidity).shape

    # Perform the transforms.
    altaz_out = sky_coords.transform_to(observing_model)

    # Warn about low altitudes when refraction is being applied.
    _warn_for_low_altitudes(altaz_out)

    if altaz_out.shape != output_shape:
        raise RuntimeError('sky_to_altaz output shape is {0} but expected {1}.'
                           .format(altaz_out.shape, output_shape))
    return altaz_out


def altaz_to_sky(alt, az, observing_model, frame='icrs'):
    """Convert (alt,az) to sky coordinates for specified observing conditions.

    Transformations between sky coordinates such as RA,DEC and ALT,AZ involve
    two steps.  First, define the atmosphere through which the sky is being
    observed with a call to :func:`create_observing_model`.  Next, call this
    function. For example, to map a North pointing at 60 degrees altitude from
    Kitt Peak onto the sky, observing at a wavelength of 5400 Angstroms:

    >>> where = observatories['KPNO']
    >>> when = astropy.time.Time('2001-01-01T00:00:00')
    >>> obs_model = create_observing_model(where, when, 5400 * u.Angstrom)
    >>> radec = altaz_to_sky(60*u.deg, 0.*u.deg, obs_model)
    >>> print('ra = %.3f deg, dec = %.3f deg' %
    ... (radec.ra.to(u.deg).value, radec.dec.to(u.deg).value))
    ra = 349.106 deg, dec = 61.962 deg

    Setting a pressure value of zero disables the atmospheric refraction model,
    so that returned coordinates are topocentric.  The atmospheric refraction
    model becomes inaccurate for altitudes below 20 degrees so a `UserWarning`
    will be issued to flag this condition.

    The output shape is determined by the usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__ applied
    to the input coordinates and the observing model input parameters.

    Parameters
    ----------
    alt : :class:`astropy.coordinates.Angle`
        Altitude angle(s) above the horizon to convert to sky coordinates.
    az : :class:`astropy.coordinates.Angle`
        Azimuthal angle(s) east of north to convert to sky coordinates.
    observing_model : :class:`astropy.coordinates.AltAz`
        The ALT-AZ coordinate frame(s) that specify the observing conditions,
        normally obtained by calling :func:`create_observing_model`.
    frame : str or :class:`astropy.coordinates.BaseCoordinateFrame` instance
        The sky coordinate frame that the input alt-az observations should be
        transformed to.  The default `'icrs'` corresponds to J2000 RA,DEC.
        Some other useful frames are `'fk5'` and `'galactic'`.  See the
        `astropy docs <http://astropy.readthedocs.io/
        en/stable/coordinates/index.html#reference-api>`__ for details.

    Returns
    -------
    :class:`astropy.coordinates.AltAz`
        An array of ALT-AZ coordinates with a shape given by
        :func:`np.broadcast(sky_coords, observing_model) <numpy.broadcast>`.
    """
    # Check that the input coordinates are compatible for broadcasting with
    # the refraction model, otherwise this will raise a ValueError.
    output_shape = np.broadcast(
        alt, az, observing_model.location, observing_model.obstime,
        observing_model.obswl, observing_model.temperature,
        observing_model.pressure, observing_model.relative_humidity).shape

    # Initialize the input coordinates.
    altaz_in = astropy.coordinates.AltAz(alt=alt, az=az,
        location=observing_model.location, obstime=observing_model.obstime,
        obswl=observing_model.obswl, temperature=observing_model.temperature,
        pressure=observing_model.pressure,
        relative_humidity=observing_model.relative_humidity)

    # Warn about low altitudes when refraction is being applied.
    _warn_for_low_altitudes(altaz_in)

    # If the frame is specified as a string, try to convert it to
    # a BaseCoordinateFrame instance.
    if isinstance(frame, basestring):
        if frame not in astropy.coordinates.frame_transform_graph.get_names():
            raise ValueError('Invalid frame name: {0}.'.format(frame))
        frame = astropy.coordinates.frame_transform_graph.lookup_name(frame)()

    # Perform the transforms.
    sky_out = altaz_in.transform_to(frame)
    if sky_out.shape != output_shape:
        raise RuntimeError('altaz_to_sky output shape is {0} but expected {1}.'
                           .format(sky_out.shape, output_shape))
    return sky_out


def adjust_time_to_hour_angle(nominal_time, target_ra, hour_angle,
                              longitude=None,
                              max_error=0.01*u.arcsec, max_iterations=3):
    """Adjust a time to a specified target hour angle.

    The input nominal time will be adjusted to the closest time where the
    specified hour angle is achieved, using either a positive or negative
    adjustment.  For example to observe Polaris from KPNO on MJD 55100 at
    its maximum elevation (hour angle = 0):

    >>> where = observatories['KPNO']
    >>> night = astropy.time.Time(55100, format='mjd', location=where)
    >>> polaris = astropy.coordinates.ICRS(ra=37.95 * u.deg, dec=89.25 * u.deg)
    >>> when = adjust_time_to_hour_angle(night, polaris.ra, 0 * u.deg)
    >>> print('MJD %.3f' % when.mjd)
    MJD 55100.401

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
    RuntimeError
        The desired accuracy could not be achieved after ``max_iterations``.
    """
    sidereal = 1 / 1.002737909350795
    when = nominal_time.copy()
    num_iterations = 0
    while True:
        with warnings.catch_warnings():
            # Ignore the UnicodeWarning that occurs in the astropy
            # implementation of sidereal_time().
            warnings.filterwarnings('ignore', 'Unicode')
            # Calculate the nominal local sidereal time of the target.
            try:
                lst = when.sidereal_time('apparent', longitude) - target_ra
            # Recent versions of astropy raise a subclass of IndexError
            # astropy.utils.iers.iers.IERSRangeError
            except IndexError:
                # Hardcode the mean UT1 - UTC offset for MJD in [54600, 57800]
                # in order to avoid issues with missing IERS tables.
                when.delta_ut1_utc = -0.1225
                lst = when.sidereal_time('apparent', longitude) - target_ra

            # Are we close enough?
            if np.abs((lst - hour_angle).wrap_at('12 hours')) <= max_error:
                break

            # Have we run out of iterations?
            if num_iterations >= max_iterations:
                raise RuntimeError(
                    'Reached max_iterations = {}.'.format(max_iterations))
            num_iterations += 1

            # Offset to the nearest time with the desired hour angle.
            # Correct for the fact that 360 deg corresponds to a sidereal day.
            when = (when - (lst - hour_angle).wrap_at('12 hours') * u.hour /
                    (15 * u.deg) * sidereal)

    return when


def _warn_for_low_altitudes(altaz):
    """Warn for low altitudes where the refraction model is inaccurate.

    This test is only performed when the pressure is non-zero, since zero
    pressure disables the atmospheric refraction model. In case both the
    altitude coordinates and the pressure values are arrays, these are
    broadcast together to test for any inacurrate (altitude, pressure)
    resulting combinations.

    Parameters
    ----------
    altaz : :class:`astropy.coordinates.AltAz`
        Input alt,az coordinates to test.
    """
    refracted = altaz.pressure.value != 0
    low = altaz.alt < low_altitude_threshold
    if isinstance(refracted, np.ndarray) and isinstance(low, np.ndarray):
        # Broadcast the pressure and altitude shapes to check if any of the
        # resulting combinations are inaccurate.
        inaccurate = np.any(np.logical_and(refracted, low))
    elif isinstance(low, np.ndarray):
        inaccurate = refracted and np.any(low)
    elif isinstance(refracted, np.ndarray):
        inaccurate = np.any(refracted) and low
    else:
        inaccurate = refracted and low
    if inaccurate:
        warnings.warn(
            'Refraction model is inaccurate for altitudes below {0}.'
            .format(low_altitude_threshold))
