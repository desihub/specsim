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
        '-n', '--num-targets', type=int, default=10, metavar='N',
        help='number of targets to simulate')
    parser.add_argument(
        '--seeing', type=float, default=1.1, metavar='FWHM',
        help='seeing FWHM at 6355A in arcseconds')
    parser.add_argument(
        '--moffat-beta', type=float, default=3.5,
        help='Moffat beta parameter for atmospheric seeing PSF')
    parser.add_argument(
        '--disk-fraction', type=float, default=1.0,
        help='Fraction of source in Sersic n=1 disk')
    parser.add_argument(
        '--bulge-fraction', type=float, default=0.0,
        help='Fraction of source in Sersic n=4 bulge')
    parser.add_argument(
        '--num-wlen', type=int, default=11, metavar='N',
        help='Number of wavelengths for interpolating fiberloss')
    parser.add_argument(
        '--num-pixels', type=int, default=16, metavar='N',
        help='number of pixels used to subdivide the fiber diameter')
    parser.add_argument(
        '--oversampling', type=int, default=32,
        help='Oversampling factor for anti-aliasing the fiber aperture')
    args = parser.parse_args(args)

    # Initialize the simulator to use.
    simulator = specsim.simulator.Simulator(args.config)

    # Initialize the wavelength grid for fiberloss calculations.
    wavelength = simulator.simulated['wavelength']
    wlen_unit = wavelength.unit
    wlen_grid = np.linspace(wavelength.data[0], wavelength.data[-1],
                            args.num_wlen) * wlen_unit

    # Calculate the seeing at each wavelength.
    simulator.atmosphere.seeing_fwhm_ref = args.seeing * u.arcsec
    seeing_fwhm = simulator.atmosphere.get_seeing_fwhm(
        wlen_grid).to(u.arcsec).value

    # Initialize a Galsim fiberloss calculator.
    calc = specsim.fiberloss.GalsimFiberlossCalculator(
        simulator.instrument.fiber_diameter.to(u.um).value,
        wlen_grid.to(u.Angstrom).value,
        args.num_pixels,
        args.oversampling,
        args.moffat_beta)

    # Generate random focal-plane coordinates for each fiber.
    gen = np.random.RandomState(seed=123)
    focal_r = (
        np.sqrt(gen.uniform(size=args.num_targets)) *
        simulator.instrument.field_radius)
    phi = 2 * np.pi * gen.uniform(size=args.num_targets)
    focal_x = np.cos(phi) * focal_r
    focal_y = np.sin(phi) * focal_r

    # Calculate optical parameters at each fiber.
    scale, blur, offset = simulator.instrument.get_focal_plane_optics(
        focal_x, focal_y, wlen_grid)

    # Generate random galaxy profiles. The two components refer to the
    # disk (Sersic n=1) and bulge (Sersic n=4).
    source_fraction = np.tile([args.disk_fraction, args.bulge_fraction],
                              (args.num_targets, 1))
    source_half_light_radius = np.tile([0.8, 1.2], (args.num_targets, 1))
    source_minor_major_axis_ratio = np.tile([0.5, 0.8], (args.num_targets, 1))
    source_position_angle = 360. * gen.uniform(size=(args.num_targets, 2))

    t_start = time.time()
    fiberloss = calc.calculate(
        seeing_fwhm,
        scale.to(u.um / u.arcsec).value, offset.to(u.um).value,
        blur.to(u.um).value,
        source_fraction, source_half_light_radius,
        source_minor_major_axis_ratio, source_position_angle)
    elapsed = time.time() - t_start

    print('Elapsed for {0} targets = {1:.3f} s, Rate = {2:.3f} ms/target'
          .format(args.num_targets, elapsed, 1e3 * elapsed / args.num_targets))
