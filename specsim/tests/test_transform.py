# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.tests.helper import pytest
from ..transform import altaz_to_focalplane, focalplane_to_altaz, \
    observatories, create_observing_model, sky_to_altaz, altaz_to_sky, \
    adjust_time_to_hour_angle

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u


def test_origin_to_focalplane():
    alt, az = 0.5 * u.rad, 1.5 * u.rad
    x, y = altaz_to_focalplane(alt, az, alt, az)
    assert np.allclose(x, 0 * u.rad) and np.allclose(y, 0 * u.rad)


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
    assert np.allclose(alt, alt0) and np.allclose(az, az0)


def test_focalplane_roundtrip():
    alt0, az0 = 0.5 * u.rad, 1.5 * u.rad
    x, y = -0.01 * u.rad, +0.02 * u.rad
    alt, az = focalplane_to_altaz(x, y, alt0, az0)
    x2, y2 = altaz_to_focalplane(alt, az, alt0, az0)
    assert np.allclose(x, x2) and np.allclose(y, y2)
    alt2, az2 = focalplane_to_altaz(x2, y2, alt0, az0)
    assert np.allclose(alt, alt2) and np.allclose(az, az2)


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
    assert np.allclose(altaz_in.alt, altaz_out.alt)
    assert np.allclose(altaz_in.az, altaz_out.az)


def test_invalid_frame():
    where = observatories['APO']
    when = Time(56383, format='mjd')
    wlen = 5400 * u.Angstrom
    obs_model = create_observing_model(where=where, when=when, wavelength=wlen)
    with pytest.raises(ValueError):
        altaz_to_sky(0.5*u.rad, 1.5*u.rad, obs_model, frame='invalid')


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
    assert np.allclose(sky_in.ra, sky_out.ra)
    assert np.allclose(sky_in.dec, sky_out.dec)


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
        adjusted = adjust_time_to_hour_angle(when, ra, 0.)
