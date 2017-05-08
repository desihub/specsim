# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

from ..simulator import *

import specsim.config

import astropy.units as u
from astropy.coordinates import SkyCoord

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


def test_ctor():
    config = specsim.config.load_config('test')
    sim1 = Simulator(config)
    sim2 = Simulator('test')
    assert sim1.atmosphere.airmass == sim2.atmosphere.airmass


def test_alt_wavelengths():
    config = specsim.config.load_config('test')
    config.wavelength_grid.step = 0.1
    config.instrument.cameras.r.constants.output_pixel_size = "1.2 Angstrom"
    sim = Simulator(config)
    sim.simulate()
    assert np.allclose(np.diff(sim.camera_output[0]['wavelength']), 1.2)

    config.instrument.cameras.r.constants.output_pixel_size = "0.4 Angstrom"
    sim = Simulator(config)
    sim.simulate()
    assert np.allclose(np.diff(sim.camera_output[0]['wavelength']), 0.4)

    config.instrument.cameras.r.constants.output_pixel_size = "0.3 Angstrom"
    sim = Simulator(config)
    sim.simulate()
    assert np.allclose(np.diff(sim.camera_output[0]['wavelength']), 0.3)

    config.wavelength_grid.step = 0.2
    config.instrument.cameras.r.constants.output_pixel_size = "0.2 Angstrom"
    sim = Simulator(config)
    sim.simulate()
    assert np.allclose(np.diff(sim.camera_output[0]['wavelength']), 0.2)

def test_end_to_end():
    sim = Simulator('test')
    sim.simulate()
    nsrc = sim.simulated['num_source_electrons_r'][:, 0].sum()
    assert np.allclose(nsrc, 86996.4478)


def test_zero_flux():
    sim = Simulator('test')
    sim.source.update_in(
        'Zero Flux', 'qso', sim.source.wavelength_in, 0 * sim.source.flux_in)
    sim.source.update_out()
    sim.simulate()
    # Check that ivar is non-zero.
    assert not np.any(sim.camera_output[0]['flux_inverse_variance'][:, 0] == 0)


def test_changing_airmass():
    sim = specsim.simulator.Simulator('test')

    sim.atmosphere.airmass = 1.0
    sim.simulate()
    sky1 = sim.camera_output[0]['num_sky_electrons'][:, 0].copy()
    sim.atmosphere.airmass = 2.0
    sim.simulate()
    sky2 = sim.camera_output[0]['num_sky_electrons'][:, 0].copy()

    sum1, sum2 = np.sum(sky1), np.sum(sky2)
    assert sum1 > 0 and sum2 > 0 and sum1 != sum2


def test_fiber_positioning():
    """Test the logic for fiber positioning.
    """
    sim = specsim.simulator.Simulator('test', num_fibers=10)
    # This should be the default.
    sim.source.focal_xy = np.array([-1, -1]) * u.mm
    sim.simulate()
    assert np.allclose(sim.focal_x.to(u.mm).value, -1.)
    assert np.allclose(sim.focal_y.to(u.mm).value, -1.)
    # Set the config sky position to the boresight.
    p = sim.observation.pointing
    sim.source.sky_position = p
    # Test it is used when x,y not set.
    sim.source.focal_xy = None
    sim.simulate()
    assert np.allclose(sim.focal_x.to(u.mm).value, 0.)
    assert np.allclose(sim.focal_y.to(u.mm).value, 0.)
    sim.source.focal_xy = np.array([-1, -1]) * u.mm
    # Build arrays to pass simulate()
    xy = np.ones((10, 2)) * u.mm
    sky = SkyCoord(ra=p.ra + np.zeros(10), dec=p.dec + np.zeros(10))
    # Check that focal_positions has the highest priority.
    sim.simulate(focal_positions=xy)
    assert np.allclose(sim.focal_x.to(u.mm).value, 1.)
    assert np.allclose(sim.focal_y.to(u.mm).value, 1.)
    sim.simulate(focal_positions=xy, sky_positions=sky)
    assert np.allclose(sim.focal_x.to(u.mm).value, 1.)
    assert np.allclose(sim.focal_y.to(u.mm).value, 1.)
    # Check that sky_positions has the next highest priority.
    sim.simulate(sky_positions=sky)
    assert np.allclose(sim.focal_x.to(u.mm).value, 0.)
    assert np.allclose(sim.focal_y.to(u.mm).value, 0.)


def test_output_table_units():
    """Test that units are preserved after calling simulate().

    This test was added in response to issue #62
    """
    sim = specsim.simulator.Simulator('test', num_fibers=1)
    units_before = {}
    for table in (sim.simulated, sim.camera_output[0]):
        for name in table.colnames:
            units_before[name] = table[name].unit
    sim.simulate()
    for table in (sim.simulated, sim.camera_output[0]):
        for name in table.colnames:
            assert units_before[name] == table[name].unit


def test_plot():
    s = Simulator('test')
    s.simulate()
    s.plot()


def test_no_cameras():
    s = Simulator('test', camera_output=False)
    s.simulate()
    s.plot()
