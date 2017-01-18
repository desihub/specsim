# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command-line script for calculating fiberloss fractions.
"""
from __future__ import print_function, division

import argparse
import time

import numpy as np

import astropy.units as u

import specsim.simulator
import specsim.fiberloss


# This is a setup.py entry-point, not a standalone script.
# See http://astropy.readthedocs.io/en/latest/development/scripts.html

def main(args=None):
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument(
        '-c', '--config', default='test',
        help='name of the simulation configuration to use')
    parser.add_argument(
        '--seeing', default=1.1, metavar='FWHM',
        help='seeing FWHM at 6355A in arcseconds')
    parser.add_argument(
        '--num-wlen', type=int, default=11, metavar='N',
        help='Number of wavelengths for interpolating fiberloss')
    parser.add_argument(
        '--num-pixels', type=int, default=32, metavar='N',
        help='number of pixels used to subdivide the fiber diameter')
    parser.add_argument(
        '-n', '--num-targets', type=int, default=5000, metavar='N',
        help='number of targets to simulate')
    args = parser.parse_args(args)

    # Build the simulator to use.
    simulator = specsim.simulator.Simulator(args.config)
    simulator.atmosphere.seeing['fwhm_ref'] = args.seeing * u.arcsec
    simulator.instrument.fiberloss_num_wlen = args.num_wlen
    simulator.instrument.fiberloss_num_pixels = args.num_pixels

    # Generate random focal-plane coordinates.
    gen = np.random.RandomState(seed=123)
    focal_r = (
        np.sqrt(gen.uniform(size=args.num_targets)) *
        simulator.instrument.field_radius)
    phi = 2 * np.pi * gen.uniform(size=args.num_targets)
    focal_x = np.cos(phi) * focal_r
    focal_y = np.sin(phi) * focal_r

    # Generate random disk and bulge position angles.
    disk_pa = 2 * np.pi * gen.uniform(size=args.num_targets) * u.rad
    bulge_pa = 2 * np.pi * gen.uniform(size=args.num_targets) * u.rad

    t_start = time.time()
    for i in xrange(args.num_targets):
        simulator.source.disk_shape.position_angle = disk_pa[i]
        simulator.source.bulge_shape.position_angle = disk_pa[i]
        fiberloss = specsim.fiberloss.calculate_fiber_acceptance_fraction(
            focal_x[i], focal_y[i], simulator.simulated['wavelength'],
            simulator.source, simulator.atmosphere, simulator.instrument)
    elapsed = time.time() - t_start

    print('Elapsed for {0} targets = {1:.3f} s, Rate = {2:.3f} ms/target'
          .format(args.num_targets, elapsed, 1e3 * elapsed / args.num_targets))
