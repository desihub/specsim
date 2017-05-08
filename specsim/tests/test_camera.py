# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..camera import *

import specsim.instrument
import specsim.config
import specsim.simulator


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


def test_output_pixel_size():
    # Reproduce the crash in https://github.com/desihub/specsim/issues/64
    config = specsim.config.load_config('test')
    dwave = 0.2
    config.wavelength_grid.min = 3554.05
    config.wavelength_grid.max = 9912.85
    config.wavelength_grid.step = dwave
    config.update()
    for n in (1, 3, 11, 100):
        size = '{0} Angstrom'.format(n)
        config.instrument.cameras.r.constants.output_pixel_size = size
        specsim.simulator.Simulator(config)
    # Check error handling for invalid output_pixel_size.
    config.instrument.cameras.r.constants.output_pixel_size = '0.3 Angstrom'
    with pytest.raises(ValueError):
        specsim.simulator.Simulator(config)
    # Check error handling for non-uniform simulation grid.
    config.instrument.cameras.r.constants.output_pixel_size = '0.2 Angstrom'
    config.wavelength[10] += 0.001 * u.Angstrom
    with pytest.raises(RuntimeError):
        specsim.simulator.Simulator(config)


def test_allow_convolution():
    c = specsim.config.load_config('test')
    i = specsim.instrument.initialize(c, camera_output=False)
    camera = i.cameras[0]
    with pytest.raises(RuntimeError):
        camera.get_output_resolution_matrix()
    with pytest.raises(RuntimeError):
        camera.downsample(None)
    with pytest.raises(RuntimeError):
        camera.apply_resolution(None)
    with pytest.raises(RuntimeError):
        s = camera.output_pixel_size
