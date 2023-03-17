# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.version import version as astropy_version
if astropy_version < '3.0':
    # With older versions of Astropy, we actually need to import the pytest
    # plugins themselves in order to make them discoverable by pytest.
    from astropy.tests.pytest_plugins import *
    
# As of Astropy 5.1 astropy.tests.plugins.display (where PYTEST_HEADER_MODULES
# and TESTED_VERSIONS lived) as been deprecated and removed entirely. 

## Uncomment the following lines to treat all DeprecationWarnings as
## exceptions
## Note that this is deprecated as of Astropy 5.1 and may be removed in the future
# from astropy.tests.helper import enable_deprecations_as_exceptions
# enable_deprecations_as_exceptions()

## Uncomment the following lines to display the version number of the
## package rather than the version number of Astropy in the top line when
## running the tests.
# import os
#
## This is to figure out the affiliated package version, rather than
## using Astropy's
# try:
#     from .version import version
# except ImportError:
#     version = 'dev'
#
# try:
#     packagename = os.path.basename(os.path.dirname(__file__))
#     TESTED_VERSIONS[packagename] = version
# except NameError:   # Needed to support Astropy <= 1.0.0
#     pass
