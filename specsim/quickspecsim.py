# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command-line script for simulating a fiber spectrograph.
"""
from __future__ import print_function, division

import warnings
import argparse

import numpy as np

import astropy.units as u

import specsim.config
import specsim.simulator


# This is a setup.py entry-point, not a standalone script.
# See http://astropy.readthedocs.io/en/latest/development/scripts.html

def main(args=None):
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('-c', '--config', default='test',
        help='name of the simulation configuration to use')
    parser.add_argument('--exposure-time', type=str, default='1000s',
        help='exposure time in to use (with units)')
    parser.add_argument('--sky-condition', type=str, default=None,
        help='sky condition to use (uses default if not set)')
    parser.add_argument('--airmass', type=float, default=1.,
        help='atmosphere airmass to use.')
    parser.add_argument('--moon-phase', type=float, default=None, metavar='P',
        help='moon phase between 0 (full) and 1 (new)')
    parser.add_argument('--moon-zenith', type=float, default=None, metavar='Z',
        help='zenith angle of the moon in degrees (>90 is below the horizon)')
    parser.add_argument('--moon-separation', type=float, default=None,
        metavar='S',
        help='opening angle between moon and this observation in degrees')
    parser.add_argument('--focal-x', type=str, default=None, metavar='X',
        help='Override x coordinate of source on focal plane (with units)')
    parser.add_argument('--focal-y', type=str, default=None, metavar='X',
        help='Override y coordinate of source on focal plane (with units)')
    parser.add_argument('--model', type=str, default=None,
        help='source fiberloss model to use (uses default if not set)')
    parser.add_argument('--z-in', type=float, default=None,
        help='redshift of input source data')
    parser.add_argument('--z-out', type=float, default=None,
        help='redshift that source should be transformed to')
    parser.add_argument('--filter', type=str, default=None,
        help='filter name to use for source flux normalization')
    parser.add_argument('--ab-mag', type=float, default=None,
        help='AB magnitude that source flux will be normalized to.')
    parser.add_argument('-o', '--output', type=str, default=None,
        help='optional output file name')
    parser.add_argument('--save-plot', type=str, default=None,
        help='save plot to the specified filename')
    parser.add_argument('--save-fiberloss', type=str, default=None,
        help='save fiberloss .fits and .ecsv files with this base filename')
    args = parser.parse_args(args)

    # Read the required configuration file.
    config = specsim.config.load_config(args.config)

    # Update configuration options from command-line options.
    config.verbose = args.verbose

    if args.sky_condition is not None:
        config.atmosphere.sky.condition = args.sky_condition
    config.atmosphere.airmass = args.airmass
    if (args.moon_phase is not None or args.moon_zenith is not None or
        args.moon_separation is not None):
        try:
            moon = config.atmosphere.moon.constants
        except AttributeError:
            print('Cannot set moon parameters when no moon defined in config.')
            return -1
        if args.moon_phase is not None:
            moon.moon_phase = args.moon_phase
        if args.moon_zenith is not None:
            moon.moon_zenith = '{0:f}deg'.format(args.moon_zenith)
        if args.moon_separation is not None:
            moon.separation_angle = '{0:f}deg'.format(args.moon_separation)

    if args.model is not None:
        config.source.type = args.model
    config.source.z_in = args.z_in
    config.source.z_out = args.z_out
    config.source.filter_name = args.filter
    config.source.ab_magnitude_out = args.ab_mag

    # Initialize the simulator.
    try:
        simulator = specsim.simulator.Simulator(config, verbose=args.verbose)
    except RuntimeError as e:
        print(e)
        return -1

    # Set parameters after configuration.
    try:
        simulator.observation.exposure_time = specsim.config.parse_quantity(
            args.exposure_time, u.s)
        if args.focal_x is not None:
            if args.focal_y is None:
                print('Must set both focal-x and focal-y.')
                return -1
            else:
                focal_x = specsim.config.parse_quantity(args.focal_x, u.mm)
                focal_y = specsim.config.parse_quantity(args.focal_y, u.mm)
                simulator.source.focal_xy = focal_x, focal_y
    except ValueError as e:
        print(e)
        return -1

    # Perform the simulation.
    simulator.simulate(save_fiberloss=args.save_fiberloss)

    # Summarize the results.
    print('Source at focal plane (x, y) = ({0:.1f}, {1:.1f}).'
          .format(simulator.focal_x[0], simulator.focal_y[0]))
    print('Observing airmass is {0:.3f}.'.format(simulator.atmosphere.airmass))
    for output in simulator.camera_output:
        camera_name = output.meta['name']
        pixel_size = output.meta['pixel_size']
        snr = (
            output['num_source_electrons'][:, 0] /
            np.sqrt(output['variance_electrons'][:, 0]))
        print('Median SNR in {0} camera = {1:.3f} / {2}'
              .format(camera_name, np.median(snr), pixel_size))

    # Save the results, if requested.
    if args.output:
        try:
            simulator.save(args.output)
        except Exception as e:
            print(e)
            return -1
        if args.verbose:
            print('Saved outputs to {0}'.format(args.output))

    # Plot the results if requested.
    if args.save_plot:
        # Defer these imports until now so that matplotlib is only required
        # if plots are requested.
        import matplotlib
        # Use a backend with minimal requirements (X11, etc).
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        simulator.plot()

        with warnings.catch_warnings():
            # Silence expected matplotlib warnings.
            warnings.simplefilter('ignore', category=FutureWarning)
            plt.savefig(args.save_plot, facecolor='white', edgecolor='none')

        if args.verbose:
            print('Saved generated plot to {0}'.format(args.save_plot))
