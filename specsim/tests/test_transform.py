# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function

from astropy.tests.helper import pytest, remote_data
from ..transform import altaz_to_focalplane, focalplane_to_altaz, \
    observatories, create_observing_model, sky_to_altaz, altaz_to_sky, \
    adjust_time_to_hour_angle, low_altitude_threshold

import warnings
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, ICRS
import astropy.units as u


def test_origin_to_focalplane():
    alt, az = 0.5 * u.rad, 1.5 * u.rad
    x, y = altaz_to_focalplane(alt, az, alt, az)
    assert np.allclose(x.to(u.rad).value, 0)
    assert np.allclose(y.to(u.rad).value, 0)


def test_focalplane_units():
    platescale = 200 * u.mm / u.deg
    alt, az = 0.5 * u.rad, 1.5 * u.rad
    x, y = altaz_to_focalplane(alt, az, alt, az, platescale=platescale)
    assert x.unit == u.m and y.unit == u.m
    alt, az = focalplane_to_altaz(x, y, alt, az, platescale=platescale)
    assert alt.unit == u.rad and az.unit == u.rad


def test_shape_to_focalplane():
    zero = 0. * u.rad
    x, y = altaz_to_focalplane(zero, zero, zero, zero)
    assert x.shape == y.shape and x.shape == ()

    angle = np.linspace(-0.1, +0.1, 3) * u.rad
    x, y = altaz_to_focalplane(angle, zero, zero, zero)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, zero, zero)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, zero, angle, zero)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, zero, zero, angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, angle, zero)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, zero, angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, zero, angle, angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(zero, angle, angle, angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, angle, angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle[:, np.newaxis], angle, zero, zero)

    assert x.shape == y.shape and x.shape == (3, 3)
    try:
        x, y = altaz_to_focalplane(angle, angle[:, np.newaxis],
            angle[:, np.newaxis, np.newaxis], zero)
        assert x.shape == y.shape and x.shape == (3, 3, 3)
        x, y = altaz_to_focalplane(angle, angle[:, np.newaxis],
            angle[:, np.newaxis, np.newaxis],
            angle[:, np.newaxis, np.newaxis, np.newaxis])
        assert x.shape == y.shape and x.shape == (3, 3, 3, 3)
        x, y = altaz_to_focalplane(angle, angle[:, np.newaxis],
            zero, angle[:, np.newaxis, np.newaxis, np.newaxis])
        assert x.shape == y.shape and x.shape == (3, 1, 3, 3)
    except RuntimeError:
        # These tests fails for numpy < 1.9 because np.einsum does not
        # broadcast correctly in this case. For details, See
        # https://github.com/desihub/specsim/issues/10
        pass


def test_focalplane_to_origin():
    alt0, az0 = 0.5 * u.rad, 1.5 * u.rad
    alt, az = focalplane_to_altaz(0. * u.rad, 0. * u.rad, alt0, az0)
    assert np.allclose(alt.to(u.rad).value, alt0.to(u.rad).value)
    assert np.allclose(az.to(u.rad).value, az0.to(u.rad).value)


def test_focalplane_roundtrip():
    alt0, az0 = 0.5 * u.rad, 1.5 * u.rad
    x, y = -0.01 * u.rad, +0.02 * u.rad
    alt, az = focalplane_to_altaz(x, y, alt0, az0)
    x2, y2 = altaz_to_focalplane(alt, az, alt0, az0)
    assert np.allclose(x.to(u.rad).value, x2.to(u.rad).value)
    assert np.allclose(y.to(u.rad).value, y2.to(u.rad).value)
    alt2, az2 = focalplane_to_altaz(x2, y2, alt0, az0)
    assert np.allclose(alt.to(u.rad).value, alt2.to(u.rad).value)
    assert np.allclose(az.to(u.rad).value, az2.to(u.rad).value)


def test_to_altaz_null():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    temperature = 5 * u.deg_C
    pressure = 800 * u.kPa
    obs_model = create_observing_model(where=where, when=when,
        wavelength=wlen, temperature=temperature, pressure=pressure)
    altaz_in = AltAz(alt=0.5*u.rad, az=1.5*u.rad, location=where,
        obstime=when, obswl=wlen, temperature=temperature, pressure=pressure)
    altaz_out = sky_to_altaz(altaz_in, obs_model)
    assert np.allclose(altaz_in.alt.to(u.rad).value,
                       altaz_out.alt.to(u.rad).value)
    assert np.allclose(altaz_in.az.to(u.rad).value,
                       altaz_out.az.to(u.rad).value)


def test_invalid_frame():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    pressure = 800 * u.kPa
    obs_model = create_observing_model(where=where, when=when, wavelength=wlen,
                                       pressure=pressure)
    with pytest.raises(ValueError):
        altaz_to_sky(0.5*u.rad, 1.5*u.rad, obs_model, frame='invalid')


@remote_data
def test_alt_no_warn():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    pressure = 0 * u.kPa
    obs_model = create_observing_model(where=where, when=when, wavelength=wlen,
                                       pressure=pressure)
    # The pytest 2.5.1 bundled with astropy_helpers does not implement
    # pytest.warns, so we turn the warning into an exception.
    warnings.simplefilter('error')
    altaz_to_sky(low_altitude_threshold - 1*u.deg, 0*u.deg, obs_model)


@remote_data
def test_alt_no_warn_pressure_array():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    pressure = np.array([0., 800.]) * u.kPa
    obs_model = create_observing_model(where=where, when=when, wavelength=wlen,
                                       pressure=pressure)
    # The pytest 2.5.1 bundled with astropy_helpers does not implement
    # pytest.warns, so we turn the warning into an exception.
    warnings.simplefilter('error')
    alt0 = low_altitude_threshold.to(u.deg).value
    alt = np.array([alt0 - 1, alt0 + 1]) * u.deg
    altaz_to_sky(alt, 0*u.deg, obs_model)


def test_alt_warn():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    pressure = 800 * u.kPa
    obs_model = create_observing_model(where=where, when=when, wavelength=wlen,
                                       pressure=pressure)
    # The pytest 2.5.1 bundled with astropy_helpers does not implement
    # pytest.warns, so we turn the warning into an exception.
    warnings.simplefilter('error')
    with pytest.raises(UserWarning):
        altaz_to_sky(low_altitude_threshold - 1*u.deg, 0*u.deg, obs_model)


def test_alt_warn_pressure_array():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    pressure = np.array([0., 800.]) * u.kPa
    obs_model = create_observing_model(where=where, when=when, wavelength=wlen,
                                       pressure=pressure)
    # The pytest 2.5.1 bundled with astropy_helpers does not implement
    # pytest.warns, so we turn the warning into an exception.
    warnings.simplefilter('error')
    alt0 = low_altitude_threshold.to(u.deg).value
    alt = np.array([alt0 + 1, alt0 - 1]) * u.deg
    with pytest.raises(UserWarning):
        altaz_to_sky(alt, 0*u.deg, obs_model)
    alt = np.array([alt0 - 1, alt0 + 1]) * u.deg
    pressure = np.array([800., 0.]) * u.kPa
    obs_model = create_observing_model(where=where, when=when, wavelength=wlen,
                                       pressure=pressure[:, np.newaxis])
    with pytest.raises(UserWarning):
        print(altaz_to_sky(alt[:, np.newaxis], 0*u.deg, obs_model).shape)


@remote_data
def test_altaz_roundtrip():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    temperature = 5 * u.deg_C
    pressure = 800 * u.kPa
    obs_model = create_observing_model(where=where, when=when,
        wavelength=wlen, temperature=temperature, pressure=pressure)
    sky_in = SkyCoord(ra=0.5*u.rad, dec=1.5*u.rad, frame='icrs')
    altaz_out = sky_to_altaz(sky_in, obs_model)
    sky_out = altaz_to_sky(altaz_out.alt, altaz_out.az, obs_model, frame='icrs')
    assert np.allclose(sky_in.ra.to(u.rad).value, sky_out.ra.to(u.rad).value)
    assert np.allclose(sky_in.dec.to(u.rad).value, sky_out.dec.to(u.rad).value)


@remote_data
def test_altaz_array_roundtrip():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    temperature = 5 * u.deg_C
    pressure = 800 * u.kPa
    obs_model = create_observing_model(where=where, when=when,
        wavelength=wlen, temperature=temperature, pressure=pressure)
    alt_in, az_in = np.linspace(20., 89., 70) * u.deg, 90. * u.deg
    sky = altaz_to_sky(alt_in, az_in, obs_model)
    altaz_out = sky_to_altaz(sky, obs_model)
    assert np.all(np.abs(altaz_out.alt - alt_in) < 0.5 * u.arcsec)
    assert np.all(np.abs(altaz_out.az - az_in) < 1e-5 * u.arcsec)


@remote_data
def test_sky_to_altaz_shape():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    temperature = 5 * u.deg_C
    pressure = 800 * u.kPa
    obs_model = create_observing_model(where=where, when=when,
        wavelength=wlen, temperature=temperature, pressure=pressure)
    angles = np.array([45., 50., 55.]) * u.deg
    assert sky_to_altaz(ICRS(ra=angles, dec=angles), obs_model).shape == (3,)
    assert sky_to_altaz(
        ICRS(ra=angles[:, np.newaxis], dec=angles), obs_model).shape == (3, 3)
    assert sky_to_altaz(
        ICRS(ra=angles, dec=angles[:, np.newaxis]), obs_model).shape == (3, 3)
    wlen = np.array([5000., 6000.]) * u.Angstrom
    obs_model = create_observing_model(where=where, when=when,
        wavelength=wlen, temperature=temperature, pressure=pressure)
    assert sky_to_altaz(ICRS(45*u.deg, 45*u.deg), obs_model).shape == (2,)
    with pytest.raises(ValueError):
        # Cannot broadcast (3,) (3,) (2,)
        sky_to_altaz(ICRS(ra=angles, dec=angles), obs_model)
    assert sky_to_altaz(
        ICRS(ra=angles[:, np.newaxis], dec=angles[:, np.newaxis]),
        obs_model).shape == (3, 2)
    assert sky_to_altaz(
        ICRS(ra=angles[:, np.newaxis, np.newaxis], dec=angles[:, np.newaxis]),
        obs_model).shape == (3, 3, 2)


@remote_data
def test_altaz_to_sky_shape():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    temperature = 5 * u.deg_C
    pressure = 800 * u.kPa
    obs_model = create_observing_model(where=where, when=when,
        wavelength=wlen, temperature=temperature, pressure=pressure)
    angles = np.array([40., 45., 50.]) * u.deg
    assert altaz_to_sky(angles, angles, obs_model).shape == (3,)
    assert altaz_to_sky(
        angles[:, np.newaxis], angles, obs_model).shape == (3, 3)
    assert altaz_to_sky(
        angles, angles[:, np.newaxis], obs_model).shape == (3, 3)
    wlen = np.array([5000., 6000.]) * u.Angstrom
    obs_model = create_observing_model(where=where, when=when,
        wavelength=wlen, temperature=temperature, pressure=pressure)
    assert altaz_to_sky(45*u.deg, 45*u.deg, obs_model).shape == (2,)
    with pytest.raises(ValueError):
        # Cannot broadcast (3,) (3,) (2,)
        altaz_to_sky(angles, angles, obs_model)
    assert altaz_to_sky(
        angles[:, np.newaxis], angles[:, np.newaxis], obs_model).shape == (3, 2)
    assert altaz_to_sky(
        angles[:, np.newaxis, np.newaxis], angles[:, np.newaxis],
        obs_model).shape == (3, 3, 2)


@remote_data
def test_adjust_null():
    ra = 45 * u.deg
    when = Time(56383, format='mjd', location=observatories['APO'])
    ha = when.sidereal_time('apparent') - ra
    adjusted = adjust_time_to_hour_angle(when, ra, ha)
    assert adjusted == when


def test_adjust_missing_longitude():
    ra = 45 * u.deg
    when = Time(56383, format='mjd', location=None)
    with pytest.raises(ValueError):
        adjusted = adjust_time_to_hour_angle(when, ra, 0. * u.deg)


@remote_data
def test_adjust_future():
    ra = 45 * u.deg
    when = Time(58000, format='mjd', location=observatories['APO'])
    adjusted = adjust_time_to_hour_angle(when, ra, 0. * u.deg)


@remote_data
def test_zero_hour_angle_adjust():
    where = observatories['KPNO']
    when = Time('2013-01-01 00:00:00', format='iso', location=where)
    adjusted = adjust_time_to_hour_angle(when, 0 * u.deg, 0. * u.deg)
