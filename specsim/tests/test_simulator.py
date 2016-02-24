# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

from ..simulator import *

import specsim.config


def test_ctor():
    config = specsim.config.load_config('test')
    sim1 = Simulator(config)
    sim2 = Simulator('test')
    assert sim1.downsampling == sim2.downsampling


def test_end_to_end():
    sim = Simulator('test')
    results = sim.simulate()
    medsnr = np.median(results[results.obsflux > 0].snrtot)
    snrtot2 = np.sum(results.snrtot ** 2)
    assert np.allclose([medsnr, snrtot2], [1.83393503, 13291.019])

'''
def test_zero_flux():
    sim = Simulator('test')
    sim.source.update_in(
        'Zero Flux', 'qso', sim.source.wavelength_in, 0 * sim.source.flux_in)
    sim.source.update_out()
    results = sim.simulate()
    # Check that ivar is non-zero.
    assert not np.all((results.camivar)[:, 0] == 0)
    assert not np.all(results.ivar == 0)
'''
