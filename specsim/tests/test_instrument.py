# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..instrument import *

import specsim.config

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


def test_resolution():
    c = specsim.config.load_config('test')
    i = initialize(c)
    R = i.cameras[0].get_output_resolution_matrix()
    np.allclose(R.sum(0)[3:-3], 1)


def test_plot():
    c = specsim.config.load_config('test')
    i = initialize(c)
    i.plot()
