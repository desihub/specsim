# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Load desimodel.fastfiberacceptance.FastFiberAcceptance.
"""
try :
    from desimodel.fastfiberacceptance import FastFiberAcceptance
except ModuleNotFoundError as e:
    print("Please update desimodel to a more recent version including desimodel.fastfiberacceptance.")
    raise(e)

print("Please consider replacing deprecated import specsim.fastfiberacceptance by desimodel.fastfiberacceptance.")
