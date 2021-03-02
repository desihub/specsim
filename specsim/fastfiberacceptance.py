import astropy.io.fits as pyfits
import numpy as np
from scipy.interpolate import RegularGridInterpolator

try :
    from desimodel.fastfiberacceptance import FastFiberAcceptance
except ModuleNotFoundError as e :
    print("Please update desimodel to a more recent version including desimodel.fastfiberacceptance.")
    raise(e)

print("Please consider replacing deprecated import specsim.fastfiberacceptance by desimodel.fastfiberacceptance.")
