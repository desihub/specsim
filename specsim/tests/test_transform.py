# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.tests.helper import pytest
from ..transform import altaz_to_focalplane

import numpy as np


def test_origin_to_focalplane():
    ra, dec = 1.23, -3.21
    x, y = altaz_to_focalplane(ra, dec, ra, dec)
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
    x, y = altaz_to_focalplane(angle[:, np.newaxis],
        angle[:, np.newaxis, np.newaxis], angle, 0.)
    assert x.shape == y.shape and x.shape == (3, 3, 3)
    x, y = altaz_to_focalplane(angle[:, np.newaxis],
        angle[:, np.newaxis, np.newaxis], angle, angle[:, np.newaxis])
    assert x.shape == y.shape and x.shape == (3, 3, 3)
