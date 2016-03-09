# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command-line script for simulating a fiber spectrograph.
"""
from __future__ import print_function, division

import os.path
import warnings

import numpy as np

from astropy.utils.compat import argparse
import astropy.units as u
import astropy.io.fits as fits

import specsim.config
import specsim.simulator


# This is a setup.py entry-point, not a standalone script.
# See http://astropy.readthedocs.org/en/latest/development/scripts.html

def main(args=None):
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('-c', '--config', default='test',
        help='name of the simulation configuration to use')
    parser.add_argument('--exposure-time', type=float, default=1000.,
        help='exposure time in seconds to use.')
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
            moon.moon_zenith = args.moon_zenith * u.deg
        if args.moon_separation is not None:
            moon.separation_angle = args.moon_separation * u.deg

    config.instrument.constants.exposure_time = (
        '{0} s'.format(args.exposure_time))

    if args.model is not None:
        config.source.type = args.model
    config.source.z_in = args.z_in
    config.source.z_out = args.z_out
    config.source.filter_name = args.filter
    config.source.ab_magnitude_out = args.ab_mag

    # Initialize the simulator.
    try:
        simulator = specsim.simulator.Simulator(config)
    except Exception as e:
        print(e)
        return -1

    # Perform the simulation.
    simulator.simulate()

    # Summarize the results.
    for output in simulator.camera_output:
        camera_name = output.meta['name']
        pixel_size = output.meta['pixel_size']
        snr = (
            output['num_source_electrons'] /
            np.sqrt(output['variance_electrons']))
        print('Median SNR in {0} camera = {1:.3f} / {2}'
              .format(camera_name, np.median(snr), pixel_size))

    # Save the results, if requested.
    if args.output:
        base, ext = os.path.splitext(args.output)
        if ext != '.fits':
            print('Output file must have the .fits extension.')
            return -1
        # Create an empty primary HDU for header keywords
        primary = fits.PrimaryHDU()
        hdr = primary.header
        hdr['config'] = args.config
        # Save each table to its own HDU.
        simulated = fits.BinTableHDU(
            name='simulated', data=simulator.simulated.as_array())
        hdus = fits.HDUList([primary, simulated])
        for output in simulator.camera_output:
            hdus.append(fits.BinTableHDU(
                name=output.meta['name'], data=output.as_array()))
        # Write the file.
        hdus.writeto(args.output, clobber=True)

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
