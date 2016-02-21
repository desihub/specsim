# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

from ..simulator import *

import specsim.config


def test_end_to_end():
    config = specsim.config.load_config('test')
    sim = Simulator(config)
    results = sim.simulate()
    medsnr = np.median(results[results.obsflux > 0].snrtot)
    snrtot2 = np.sum(results.snrtot ** 2)
    assert np.allclose([medsnr, snrtot2], [2.10661167, 18068.7423])

'''
def test_zero_flux():
    config = specsim.config.load_config('test')
    sim = Simulator(config)
    sim.source.update_in(
        'Zero Flux', 'qso', sim.source.wavelength_in, 0 * sim.source.flux_in)
    sim.source.update_out()
    results = sim.simulate()
    # Check that ivar is non-zero.
    assert not np.all((results.camivar)[:, 0] == 0)
    assert not np.all(results.ivar == 0)
'''
