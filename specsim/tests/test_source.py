# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..source import *
from ..config import load_config


def test_ctor():
    config = load_config('test')
    src = initialize(config)

    assert np.array_equal(src.wavelength_in.value, [1000., 20000.])
    assert np.array_equal(src.flux_in.value, [1e-17, 1e-17])
    assert src.wavelength_in.unit == u.Angstrom
    assert src.flux_in.unit == u.erg / (u.cm ** 2 * u.s * u.Angstrom)

    assert np.array_equal(src.wavelength_out.value, config.wavelength)
    assert np.all(src.flux_out.value == 1e-17)
    assert src.flux_out.unit == src.flux_in.unit


def test_updates():
    config = load_config('test')
    src = initialize(config)

    src.update_in('name', 'type_name', src.wavelength_in, 2 * src.flux_in)
    src.update_out()
    assert np.all(src.flux_out.value == 2e-17)


def test_update_interlock():
    config = load_config('test')
    src = initialize(config)

    src.update_in('name', 'type_name', src.wavelength_in, src.flux_in)
    with pytest.raises(RuntimeError):
        f = src.flux_out


def test_redshift():
    config = load_config('test')
    src = initialize(config)

    with pytest.raises(RuntimeError):
        src.update_out(z_out = 0.5)
    src.update_in('name', 'type_name', src.wavelength_in, src.flux_in, z_in=0.)
    src.update_out(z_out = 0.5)
    assert np.all(src.flux_out.value == 1e-17 / (1 + 0.5))


def test_normalize():
    config = load_config('test')
    src = initialize(config)
    src.update_out(filter_name='sdss2010-r', ab_magnitude_out=22.)

    rband = speclite.filters.load_filter('sdss2010-r')
    abmag = rband.get_ab_magnitude(src.flux_out, src.wavelength_out)
    assert abs(abmag - 22.) < 1e-8


def test_side_effects():
    config = load_config('test')
    src = initialize(config)
    src.update_in('name', 'type_name', src.wavelength_in, src.flux_in, z_in=0.)
    wave_in = src.wavelength_in.copy()
    flux_in = src.flux_in.copy()
    src.update_out(z_out=0.5)
    assert np.all(wave_in == src.wavelength_in)
    assert np.all(flux_in == src.flux_in)
