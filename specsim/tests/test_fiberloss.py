# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest

import numpy as np
import astropy.units as u
import astropy.utils
import specsim.simulator

from ..fiberloss import *

def test_fiberloss():
    sim = specsim.simulator.Simulator('test', num_fibers=1)
    xy = np.array([0.,]) * u.mm
    wlen = np.linspace(4000., 10000., 7) * u.Angstrom
    floss = calculate_fiber_acceptance_fraction(
        xy, xy, wlen, sim.source, sim.atmosphere, sim.instrument)
    assert(np.allclose(np.mean(floss[0]), 0.5500))

def test_galsim():
    try:
        import galsim
    except ImportError:
        return
    sim = specsim.simulator.Simulator('test', num_fibers=1)
    sim.instrument.fiberloss_method = 'galsim'
    xy = np.array([0.,]) * u.mm
    wlen = np.linspace(4000., 10000., 7) * u.Angstrom
    floss = calculate_fiber_acceptance_fraction(
        xy, xy, wlen, sim.source, sim.atmosphere, sim.instrument)
    assert(np.allclose(np.mean(floss[0]), 0.5653))
