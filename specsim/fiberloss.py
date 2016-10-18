# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate fiberloss fractions.

Fiberloss fractions are computed as the overlap between the light profile
illuminating a fiber and the on-sky aperture of the fiber.
"""
from __future__ import print_function, division


def calculate_fiber_acceptance_fraction(source, atmosphere, instrument,
                                        observation):
    """
    """
    return instrument.get_fiber_acceptance(source)
