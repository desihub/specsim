# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..instrument import *

import specsim.config

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


def test_plot():
    c = specsim.config.load_config('test')
    i = initialize(c)
    i.plot()


def test_distortion_plot():
    c = specsim.config.load_config('test')
    i = initialize(c)
    i.plot_field_distortion()


def test_no_cameras():
    c = specsim.config.load_config('test')
    i = initialize(c, camera_output=False)
    i.plot()
