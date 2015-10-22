# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.tests.helper import pytest
from ..transform import altaz_to_focalplane, focalplane_to_altaz

import numpy as np


def test_origin_to_focalplane():
    alt, az = 0.5, 1.5
    x, y = altaz_to_focalplane(alt, az, alt, az)
    assert np.allclose([x, y], [0, 0])


def test_shape_to_focalplane():
    x, y = altaz_to_focalplane(0., 0., 0., 0.)
    assert x.shape == y.shape and x.shape == ()

    angle = np.linspace(-0.1, +0.1, 3)
    x, y = altaz_to_focalplane(angle, 0., 0., 0.)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, 0., 0.)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, 0., angle, 0.)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, 0., 0., angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, angle, 0.)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, 0., angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, 0., angle, angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(0., angle, angle, angle)
    assert x.shape == y.shape and x.shape == (3,)
    x, y = altaz_to_focalplane(angle, angle, angle, angle)
    assert x.shape == y.shape and x.shape == (3,)

    x, y = altaz_to_focalplane(angle[:, np.newaxis], angle, 0., 0.)
    assert x.shape == y.shape and x.shape == (3, 3)
    x, y = altaz_to_focalplane(angle, angle[:, np.newaxis],
        angle[:, np.newaxis, np.newaxis], 0.)
    assert x.shape == y.shape and x.shape == (3, 3, 3)


def test_focalplane_to_origin():
    alt0, az0 = 0.5, 1.5
    alt, az = focalplane_to_altaz(0., 0., alt0, az0)
    assert np.allclose([alt, az], [alt0, az0])


def test_focalplane_roundtrip():
    alt0, az0 = 0.5, 1.5
    x, y = -0.01, +0.02
    alt, az = focalplane_to_altaz(x, y, alt0, az0)
    x2, y2 = altaz_to_focalplane(alt, az, alt0, az0)
    assert np.allclose([x, y], [x2, y2])
    alt2, az2 = focalplane_to_altaz(x2, y2, alt0, az0)
    assert np.allclose([alt, az], [alt2, az2])
