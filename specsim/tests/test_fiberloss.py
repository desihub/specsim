# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test specsim.fiberloss.
"""
import pytest
import numpy as np
import astropy.units as u
import specsim.simulator

from ..fiberloss import *

galsim_installed = True
try:
    import galsim
except ImportError:
    galsim_installed = False


def test_fiberloss():
    sim = specsim.simulator.Simulator('test', num_fibers=1)
    xy = np.array([0.,]) * u.mm
    wlen = np.linspace(4000., 10000., 7) * u.Angstrom
    floss = calculate_fiber_acceptance_fraction(
        xy, xy, wlen, sim.source, sim.atmosphere, sim.instrument)
    assert(np.allclose(np.mean(floss[0]), 0.5500))


@pytest.mark.skipif(not galsim_installed, reason="The galsim package is not installed.")
def test_galsim():
    sim = specsim.simulator.Simulator('test', num_fibers=1)
    sim.instrument.fiberloss_method = 'galsim'
    xy = np.array([0.,]) * u.mm
    wlen = np.linspace(4000., 10000., 7) * u.Angstrom
    floss = calculate_fiber_acceptance_fraction(
        xy, xy, wlen, sim.source, sim.atmosphere, sim.instrument)
    assert(np.allclose(np.mean(floss[0]), 0.5653))
