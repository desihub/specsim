# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest

import specsim.simulator

from ..fiberloss import *


def test_fiberloss():
    sim = specsim.simulator.Simulator('test')
    fa = calculate_fiber_acceptance_fraction(
        0. * u.mm, 0. * u.mm, 6000 * u.Angstrom,
        sim.source, sim.atmosphere, sim.instrument)
