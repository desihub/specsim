# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function

from astropy.tests.helper import pytest, remote_data

from ..observation import *
from ..instrument import initialize as instrument_init
from ..config import load_config

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def test_locate():
    config = load_config('test')
    obs = initialize(config)
    ins = instrument_init(config)
    x, y = obs.locate_on_focal_plane(obs.pointing, ins)
    assert np.allclose([x.to(u.mm).value, y.to(u.mm).value], [0, 0])
    offset = SkyCoord(ra=obs.pointing.ra + 1 * u.deg,
                      dec=obs.pointing.dec - 1 * u.deg, frame='icrs')
    x, y = obs.locate_on_focal_plane(offset, ins)
    assert np.allclose([x.to(u.mm).value, y.to(u.mm).value],
                       [-196.89256, -299.27620])
