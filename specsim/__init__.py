# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Quick simulations of fiber spectrograph response
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .spectrum import WavelengthFunction,SpectralFluxDensity
    from .instrument import Instrument
    from .atmosphere import Atmosphere
    from .quick import Quick
