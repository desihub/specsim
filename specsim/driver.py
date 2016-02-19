# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Quick simulator of a fiber spectrograph.

Use quickspecsim --help for instructions on running this program.
Based on the IDL code pro/desi_quicksim.pro by D. Schlegel (LBL)

The SPECSIM_MODEL environment variable should be set to the name of a
directory containing the configuration files for the simulator.
For DESI simulations, use the top-level directory of the desimodel package.

Created 23-Jun-2014 by David Kirkby (dkirkby@uci.edu)
"""
from __future__ import print_function, division

import os
import os.path

import numpy as np

from astropy.utils.compat import argparse

import specsim


# This is a setup.py entry-point, not a standalone script.
# See http://astropy.readthedocs.org/en/latest/development/scripts.html

def main(args=None):
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v','--verbose', action = 'store_true',
        help = 'provide verbose output on progress')
    parser.add_argument('-c', '--config', default = 'test',
        help = 'name of the simulation configuration to use')
    parser.add_argument('--exposure-time', type=float, default=1000.,
        help = 'exposure time in seconds to use.')
    parser.add_argument('--sky-condition', type=str, default=None,
        help = 'sky condition to use (uses default if not set)')
    parser.add_argument('--airmass', type=float, default=1.,
        help = 'atmosphere airmass to use.')
    parser.add_argument('--model', type=str, default=None,
        help = 'source fiberloss model to use (uses default if not set)')
    parser.add_argument('--z-in', type=float, default=None,
        help = 'redshift of input source data')
    parser.add_argument('--z-out', type=float, default=None,
        help = 'redshift that source should be transformed to')
    parser.add_argument('--filter', type=str, default=None,
        help = 'filter name to use for source flux normalization')
    parser.add_argument('--ab-mag', type=float, default=None,
        help = 'AB magnitude that source flux will be normalized to.')
    parser.add_argument('--outfile', type = str, default = None,
        help = 'optional output file name')
    parser.add_argument('--show-plot', action = 'store_true',
        help = 'display the simulated spectrum and wait until plot window is closed')
    parser.add_argument('--save-plot', type = str, default = None,
        help = 'save plot to the specified filename')
    parser.add_argument('--plot-min', type = float, default = None,
        help = 'minimum wavelength to include in plot (default is full range)')
    parser.add_argument('--plot-max', type = float, default = None,
        help = 'maximum wavelength to include in plot (default is full range)')
    args = parser.parse_args(args)

    # Read the required configuration file.
    config = specsim.config.load_config(args.config)

    # Update configuration options from command-line options.
    config.verbose = args.verbose

    if args.sky_condition is not None:
        config.atmosphere.sky.condition = args.sky_condition
    config.atmosphere.airmass = args.airmass

    config.instrument.constants.exposure_time = (
        '{0} s'.format(args.exposure_time))

    if args.model is not None:
        config.source.type = args.model
    config.source.z_in = args.z_in
    config.source.z_out = args.z_out
    config.source.filter_name = args.filter
    config.source.ab_magnitude = args.ab_mag
    specSummary = config.source.name

    # Initialize the source to simulate.
    source = specsim.source.initialize(config)

    # Initialize the simulator.
    simulator = specsim.simulator.Simulator(config)

    # Perform a quick simulation of the observed spectrum.
    results = simulator.simulate(source)

    # Calculate the median total SNR in bins with some observed flux.
    medianSNR = np.median(results[results.obsflux > 0].snrtot)
    # Calculate the total SNR^2 for the combined cameras.
    totalSNR2 = np.sum(results.snrtot**2)
    # Print a summary of SNR statistics.
    snrSummary = 'Median S/N = %.3f, Total (S/N)^2 = %.1f' % (medianSNR,totalSNR2)
    print(snrSummary)

    # Save the results if requested (using the same format as the original IDL code).
    if args.outfile:
        if args.verbose:
            print('Saving results to %s' % args.outfile)
        # Calculate the total number of observed source and sky photons in all cameras.
        nobj = np.sum(results.nobj,axis=1)
        nsky = np.sum(results.nsky,axis=1)
        # Try opening the requested output file.
        with open(args.outfile,'w') as out:
            print('# AIRMASS=',simulator.airmass,file=out)
            print('# MODEL=', source.type_name, file=out)
            print('# EXPTIME=',simulator.expTime,file=out)
            print('#',file=out)
            print('# Median (S/N)=',medianSNR,file=out)
            print('# Total (S/N)^2=',totalSNR2,file=out)
            print('#',file=out)
            print('# Wave    Flux        Invvar      S/N         Counts_obj  Counts_sky  Counts_read  FWHM',
                file=out)
            print('# [Ang] [e-17 erg/s/cm^2/Ang] [1/(e-17 erg/s/cm^2/Ang)^2] []',
                '[electrons] [electrons] [electrons] [Ang]',file=out)
            for i,row in enumerate(results):
                rdnoise,psf = 0,0
                print('%9.2f %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %7.3f'
                    % (row.wave,row.obsflux,row.ivar,row.snrtot,nobj[i],nsky[i],rdnoise,psf),file=out)

    # Plot the results if requested.
    if args.show_plot or args.save_plot:
        # Defer these imports until now so that matplotlib is only required if plots are requested.
        import matplotlib
        if not args.show_plot:
            # Use a backend with minimal requirements (X11, etc).
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Make the plots, labeled with our SNR summary.
        simulator.plot(results,labels=(specSummary,snrSummary),
                       plotMin=args.plot_min,plotMax=args.plot_max)
        # Save the plot if requested.
        if args.save_plot:
            plt.savefig(args.save_plot)
            if args.verbose:
                print('Saved generated plot to',args.save_plot)
        # Show the plot interactively if requested.
        if args.show_plot:
            print('Close the plot window to exit...')
            plt.show()
