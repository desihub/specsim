# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..atmosphere import *

import specsim.config

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


def test_initialize():
    c = specsim.config.load_config('test')
    a = initialize(c)
    assert a.airmass == 1.0
    assert a.moon.moon_phase == 0.5


def test_read_only_properties():
    c = specsim.config.load_config('test')
    a = initialize(c)
    with pytest.raises(AttributeError):
        a.moon = None
    with pytest.raises(AttributeError):
        a.condition_names = None
    with pytest.raises(AttributeError):
        a.surface_brightness = None
    with pytest.raises(AttributeError):
        a.extinction = None
    m = a.moon
    with pytest.raises(AttributeError):
        m.surface_brightness = None
    with pytest.raises(AttributeError):
        m.visible = None
    with pytest.raises(AttributeError):
        m.obs_zenith = None
    with pytest.raises(AttributeError):
        m.vband_extinction = None


def test_property_updates():
    c = specsim.config.load_config('test')
    a = initialize(c)
    m = a.moon

    assert m._update_required == True
    assert np.allclose(
        np.mean(a.surface_brightness.value), 1.563611e-17, atol=0.)
    assert m.visible == True
    assert np.allclose(m.obs_zenith.value, 0.)
    # Evaluating the atmosphere surface_brightness updates the moon.
    assert m._update_required == False
    assert np.allclose(
        np.mean(m.surface_brightness.value), 6.370824e-18, atol=0.)

    # Changing the atmosphere airmass invalidates the moon surface brightness.
    a.airmass = 2.5
    assert m._update_required == True
    assert np.allclose(m.obs_zenith.value, 1.20942920)
    assert np.allclose(
        np.mean(a.surface_brightness.value), 2.198837e-17, atol=0.)
    assert np.allclose(
        np.mean(m.surface_brightness.value), 1.430046e-17, atol=0.)
    assert m._update_required == False

    a.airmass = 1.0
    assert np.allclose(
        np.mean(a.surface_brightness.value), 1.563611e-17, atol=0.)
    assert np.allclose(
        np.mean(m.surface_brightness.value), 6.370824e-18, atol=0.)
    assert np.allclose(m.obs_zenith.value, 0.)


def test_seeing():
    c = specsim.config.load_config('test')
    a = initialize(c)
    # No setters for moffat_beta and wlen_ref
    with pytest.raises(AttributeError):
        a.seeing_moffat_beta = None
    with pytest.raises(AttributeError):
        a.seeing_wlen_ref = None
    # Units must be ok when setting fwhm_ref
    a.seeing_fwhm_ref = 1.0 * u.arcsec
    with pytest.raises(ValueError):
        a.seeing_fwhm_ref = 1.0
    with pytest.raises(ValueError):
        a.seeing_fwhm_ref = 1.0 * u.m


def test_seeing_none():
    c = specsim.config.load_config('test')
    a = initialize(c)
    a._seeing = None
    assert a.seeing_moffat_beta is None
    assert a.seeing_wlen_ref is None
    assert a.seeing_fwhm_ref is None
    with pytest.raises(ValueError):
        a.seeing_fwhm_ref = 1.5 * u.arcsec


def test_plot():
    c = specsim.config.load_config('test')
    a = initialize(c)
    a.plot()


def test_plot_moon():
    plot_lunar_brightness(
        moon_zenith=60*u.deg, moon_azimuth=90*u.deg, moon_phase=0.25)
