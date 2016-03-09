# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

from ..simulator import *

import specsim.config

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


def test_ctor():
    config = specsim.config.load_config('test')
    sim1 = Simulator(config)
    sim2 = Simulator('test')
    assert sim1.atmosphere.airmass == sim2.atmosphere.airmass


def test_end_to_end():
    sim = Simulator('test')
    sim.simulate()
    nsrc = sim.simulated['num_source_electrons_r'].sum()
    assert np.allclose(nsrc, 86996.4478)


def test_zero_flux():
    sim = Simulator('test')
    sim.source.update_in(
        'Zero Flux', 'qso', sim.source.wavelength_in, 0 * sim.source.flux_in)
    sim.source.update_out()
    sim.simulate()
    # Check that ivar is non-zero.
    assert not np.any(sim.camera_output[0]['flux_inverse_variance'] == 0)

def test_plot():
    s = Simulator('test')
    s.simulate()
    s.plot()
