# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest

from ..fiberloss import *


def test_fiberloss():
    calculate_fiberloss_fraction()
