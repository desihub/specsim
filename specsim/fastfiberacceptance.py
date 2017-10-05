import astropy.io.fits as pyfits
import numpy as np
from scipy.interpolate import RegularGridInterpolator



class FastFiberAcceptance(object):
    """
    This class reads an input fits file generated with specsim.fitgalsim
    ($DESIMODEL/data/throughput/galsim-fiber-acceptance.fits)
    and instanciates RegularGridInterpolator objects for 2D and 3D interpolation
    of the pre-computed galsim fiber acceptance as a function of sigma (atmosphere+telescope blur, in um on focal surface), fiber offset from source (in um on focal surface), and half light radius (in arcsec) from extended source.
    The average and rms interpolation function for POINT,DISK and BULGE profiles
    are loaded.
    """
    def __init__(self,filename):
        
        hdulist=pyfits.open(filename)
        
        sigma=hdulist["SIGMA"].data
        offset=hdulist["OFFSET"].data
        hlradius=hdulist["HLRAD"].data
        
        self.fiber_acceptance_func = {}
        self.fiber_acceptance_rms_func = {}
        for source in ["POINT","DISK","BULGE"] :

            data=hdulist[source].data
            rms=hdulist[source[0]+"RMS"].data
            dim=len(data.shape)
            if dim == 2 :
                self.fiber_acceptance_func[source] = RegularGridInterpolator(points=(sigma,offset),values=data,method="linear",bounds_error=False,fill_value=None)
                self.fiber_acceptance_rms_func[source] = RegularGridInterpolator(points=(sigma,offset),values=rms,method="linear",bounds_error=False,fill_value=None)
            elif dim == 3 :
                self.fiber_acceptance_func[source] = RegularGridInterpolator(points=(hlradius,sigma,offset),values=data,method="linear",bounds_error=False,fill_value=None)
                self.fiber_acceptance_rms_func[source] = RegularGridInterpolator(points=(hlradius,sigma,offset),values=rms,method="linear",bounds_error=False,fill_value=None)

        hdulist.close()
    
    def rms(self,source,sigmas,offsets,hlradii=None) :
        """
        returns fiber acceptance fraction rms for the given source,sigmas,offsets
        
        Args:
            source (string) : POINT, DISK or BULGE for point source, exponential profile or De Vaucouleurs profile
            sigmas (np.array) : arbitrary shape, values of sigmas in um for the PSF due to atmosphere and telescope blur
            offsets (np.array) : same shape as sigmas, values of offsets on focal surface between fiber and source, in um
        
        Optional:
            hlradii (np.array) : same shape as sigmas, half light radius in arcsec for source

        Returns np.array with same shape as input
        """

        assert(sigmas.shape==offsets.shape)
        if hlradii is not None :
            assert(hlradii.shape==offsets.shape)
        
        original_shape = sigmas.shape

        res = None
        if source == "POINT" :
            
            res = self.fiber_acceptance_rms_func[source](np.array([sigmas.ravel(),offsets.ravel()]).T)
                    
        else :
            
            if hlradii is None :
                if source == "DISK" :
                    hlradii = 0.45 * np.ones(sigmas.shape)
                elif source == "BULGE" :
                    hlradii = 1. * np.ones(sigmas.shape)
            res = self.fiber_acceptance_rms_func[source](np.array([hlradii.ravel(),sigmas.ravel(),offsets.ravel()]).T)
        
        res[res<0] = 0.
        res[res>1] = 1.
        return res.reshape(original_shape)
    
    def value(self,source,sigmas,offsets,hlradii=None) :
        """
        returns the fiber acceptance for the given source,sigmas,offsets
        
        Args:
            source (string) : POINT, DISK or BULGE for point source, exponential profile or De Vaucouleurs profile
            sigmas (np.array) : arbitrary shape, values of sigmas in um for the PSF due to atmosphere and telescope blur
            offsets (np.array) : same shape as sigmas, values of offsets on focal surface between fiber and source, in um
        
        Optional:
            hlradii (np.array) : same shape as sigmas, half light radius in arcsec for source

        Returns np.array with same shape as input
        """
        
        assert(sigmas.shape==offsets.shape)
        if hlradii is not None :
            assert(hlradii.shape==offsets.shape)
        
        original_shape = sigmas.shape
        
        res = None
        if source == "POINT" :

            res = self.fiber_acceptance_func[source](np.array([sigmas.ravel(),offsets.ravel()]).T)

        else :

            if hlradii is None :
                if source == "DISK" :
                    hlradii = 0.45 * np.ones(sigmas.shape)
                elif source == "BULGE" :
                    hlradii = 1. * np.ones(sigmas.shape)
            
            res = self.fiber_acceptance_func[source](np.array([hlradii.ravel(),sigmas.ravel(),offsets.ravel()]).T)
        
        res[res<0] = 0.
        res[res>1] = 1.
        return res.reshape(original_shape)
        
