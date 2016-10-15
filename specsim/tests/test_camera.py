# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..camera import *

import specsim.instrument
import specsim.config


def test_resolution():
    c = specsim.config.load_config('test')
    i = specsim.instrument.initialize(c)
    R = i.cameras[0].get_output_resolution_matrix()
    assert np.allclose(R.sum(0)[3:-3], 1)


def test_downsampling():
    c = specsim.config.load_config('test')
    i = specsim.instrument.initialize(c)
    camera = i.cameras[0]

    # Use an intermediate dense matrix for downsampling.
    # This is the old implementation of get_output_resolution_matrix()
    # which uses too much memory.
    n = len(camera._output_wavelength)
    m = camera._downsampling
    i0 = camera.ccd_slice.start - camera.response_slice.start
    R1 = (camera._resolution_matrix[: n * m, i0 : i0 + n * m].toarray()
         .reshape(n, m, n, m).sum(axis=3).sum(axis=1) / float(m))

    # Use the new sparse implementation of get_output_resolution_matrix().
    R2 = camera.get_output_resolution_matrix()

    assert np.allclose(R1, R2.toarray())
