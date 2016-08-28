# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest

from ..config import *


def test_bad_name(tmpdir):
    name = str(tmpdir.join('_non_existent_.dat'))
    with pytest.raises(ValueError):
        load_config(name)
    name = str(tmpdir.join('_non_existent_.yaml'))
    with pytest.raises(ValueError):
        load_config(name)


def test_bad_key_type(tmpdir):
    name = str(tmpdir.join('test.yaml'))
    with open(name, 'w') as f:
        f.write('[1, 2]: 3')
    with pytest.raises(RuntimeError):
        load_config(name)


def test_bad_key_value(tmpdir):
    name = str(tmpdir.join('test.yaml'))
    with open(name, 'w') as f:
        f.write('bad-key: 0')
    with pytest.raises(RuntimeError):
        load_config(name)


def test_sequence(tmpdir):
    name = str(tmpdir.join('test.yaml'))
    with open(name, 'w') as f:
        f.write('key: [1, 2]')
    with pytest.raises(RuntimeError):
        load_config(name)


def test_node_getattr(tmpdir):
    name = str(tmpdir.join('test.yaml'))
    with open(name, 'w') as f:
        f.write('a: { b: 1, c: {d: 2, e: three}}')
    config = load_config(name, config_type=Node)
    assert isinstance(config, Node)
    assert isinstance(config.a, Node)
    assert isinstance(config.a.c, Node)
    assert str(config) == ''
    assert str(config.a) == 'a'
    assert str(config.a.c) == 'a.c'
    assert config.a.b == 1
    assert config.a.c.d == 2
    assert config.a.c.e == 'three'


def test_node_setattr(tmpdir):
    name = str(tmpdir.join('test.yaml'))
    with open(name, 'w') as f:
        f.write('a: { b: 1, c: {d: 2, e: three}}')
    config = load_config(name, config_type=Node)
    config.a.b = 1.23
    assert config.a.b == 1.23


def test_constants():
    config = load_config('test')
    const = config.get_constants(config.instrument.cameras.r)
    assert const['dark_current'] == 2.0 * u.electron / (u.hour * u.pixel**2)


def test_parse_quantity():
    assert parse_quantity('1') == u.Quantity(1)
    assert parse_quantity('1.23') == u.Quantity(1.23)
    assert parse_quantity('1.23m') == 1.23 * u.m
    assert parse_quantity('1.23 m') == 1.23 * u.m
    assert parse_quantity('1.23 m/s') == 1.23 * u.m / u.s
    assert parse_quantity('1.23 m / s') == 1.23 * u.m / u.s
    with pytest.raises(ValueError):
        parse_quantity('123 abc')
    with pytest.raises(ValueError):
        parse_quantity('m/s')


def test_parse_dimensions():
    assert parse_quantity('1min', 's') == 60 * u.s
    assert parse_quantity('1min', u.s) == 60 * u.s
    with pytest.raises(ValueError):
        parse_quantity('1m', u.s)


def test_table():
    config = load_config('test')
    t = config.load_table(config.source, 'flux')
    assert len(t) == len(config.wavelength)
    assert t.unit == u.erg / (u.cm**2 * u.s * u.Angstrom)
