# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..camera import *

import specsim.instrument
import specsim.config


def test_resolution():
    c = specsim.config.load_config('test')
    i = specsim.instrument.initialize(c)
    R = i.cameras[0].get_output_resolution_matrix()
    np.allclose(R.sum(0)[3:-3], 1)
