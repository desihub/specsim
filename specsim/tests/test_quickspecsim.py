# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from ..quickspecsim import main


def test_quickspecsim(tmpdir):
    save = str(tmpdir.join('test.dat'))
    main('-c test -o {0}'.format(save).split())
