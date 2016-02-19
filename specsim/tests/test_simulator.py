# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..simulator import *

import specsim.config

def test_end_to_end():
    config = specsim.config.load_config('test')
    sim = Simulator(config)
    results = sim.simulate()
    medsnr = np.median(results[results.obsflux > 0].snrtot)
    snrtot2 = np.sum(results.snrtot ** 2)
    assert np.allclose([medsnr, snrtot2], [2.10661167, 18068.7423])
