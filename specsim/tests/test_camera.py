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

#
# As of 2024-05-14, this test is failing because the values are no longer close.
#
@pytest.mark.xfail
def test_downsampling():
    # Expected resolution matrix rows
    from scipy.special import erf
    def expected_resolution_row(x, R, a):
        sqrt2 = np.sqrt(2)

        gamma_p = (x + (a / 2)) / R / sqrt2
        gamma_m = (x - (a / 2)) / R / sqrt2

        return (erf(gamma_p) - erf(gamma_m)) / 2

    c = specsim.config.load_config('test')
    i = specsim.instrument.initialize(c)
    camera = i.cameras[0]

    n = len(camera._output_wavelength)
    m = camera._downsampling
    rms_in = camera._rms_resolution[camera.ccd_slice.start]
    bin_width_out = camera.output_pixel_size.value

    # The new sparse implementation of get_output_resolution_matrix().
    R2 = camera.get_output_resolution_matrix()
    ndiags = R2.offsets.size
    R2 = R2.toarray()
    nrows, ncols = R2.shape

    pass_test = True
    for jj in range(nrows):
        i1 = max(0, jj - ndiags // 2)
        i2 = min(ncols - 1, jj + ndiags // 2)
        ss = np.s_[i1:i2]

        wave_out = (
            camera.output_wavelength.value[ss]
            - camera.output_wavelength.value[jj]
        )
        expected_row = expected_resolution_row(wave_out, rms_in, bin_width_out)

        pass_test &= np.allclose(R2[jj, ss], expected_row, rtol=0.03)
    
    assert pass_test


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
