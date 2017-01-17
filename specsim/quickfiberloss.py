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
        '-n', '--num-targets', type=int, default=5000, metavar='N',
        help='number of targets to simulate')
    args = parser.parse_args(args)

    # Build the simulator to use.
    simulator = specsim.simulator.Simulator(args.config)
    simulator.atmosphere.seeing['fwhm_ref'] = args.seeing * u.arcsec

    # Generate random focal-plane coordinates.
    gen = np.random.RandomState(seed=123)
    focal_r = (
        np.sqrt(gen.uniform(size=args.num_targets)) *
        simulator.instrument.field_radius)
    phi = gen.uniform(0., 2 * np.pi, size=args.num_targets)
    focal_x = np.cos(phi) * focal_r
    focal_y = np.sin(phi) * focal_r

    #np.save('fx.npy', focal_x.value)
    #np.save('fy.npy', focal_y.value)

    t_start = time.time()
    for i in xrange(args.num_targets):
        fiberloss = specsim.fiberloss.calculate_fiber_acceptance_fraction(
            focal_x[i], focal_y[i], simulator.simulated['wavelength'],
            simulator.source, simulator.atmosphere, simulator.instrument)
    elapsed = 1e6 * (time.time() - t_start)

    print('Elapsed for {0} targets = {1:.3f} us, Rate = {2:.3f} us/target'
          .format(args.num_targets, elapsed, elapsed / args.num_targets))
