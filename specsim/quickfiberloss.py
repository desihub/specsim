# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command-line script for calculating fiberloss fractions.
"""
from __future__ import print_function, division

import os.path
import warnings
import argparse

import numpy as np

import astropy.units as u
import astropy.io.fits as fits

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
    args = parser.parse_args(args)

    # Read the required configuration file.
    config = specsim.config.load_config(args.config)

    # Update configuration options from command-line options.
    config.verbose = args.verbose
