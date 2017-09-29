import astropy.io.fits as pyfits
import numpy as np
from scipy.interpolate import RegularGridInterpolator



class FastFiberAcceptance(object):
    
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
        Returns the uncertainty or rms of fiber acceptance for the given source,sigmas,offsets
        """

        assert(sigmas.shape==offsets.shape)
        if hlradii is not None :
            assert(hlradii.shape==offsets.shape)
        
        original_shape = sigmas.shape

        if source == "POINT" :
            
            return self.fiber_acceptance_rms_func[source](np.array([sigmas.ravel(),offsets.ravel()]).T).reshape(original_shape)
        
        else :
            
            if hlradii is None :
                if source == "DISK" :
                    hlradii = 0.45 * np.ones(sigmas.shape)
                elif source == "BULGE" :
                    hlradii = 1. * np.ones(sigmas.shape)
            return self.fiber_acceptance_rms_func[source](np.array([hlradii.ravel(),sigmas.ravel(),offsets.ravel()]).T).reshape(original_shape)
    
    def value(self,source,sigmas,offsets,hlradii=None) :
        """
        Returns the fiber acceptance for the given source,sigmas,offsets
        """
        
        assert(sigmas.shape==offsets.shape)
        if hlradii is not None :
            assert(hlradii.shape==offsets.shape)
        
        original_shape = sigmas.shape

        if source == "POINT" :

            return self.fiber_acceptance_func[source](np.array([sigmas.ravel(),offsets.ravel()]).T).reshape(original_shape)

        else :

            if hlradii is None :
                if source == "DISK" :
                    hlradii = 0.45 * np.ones(sigmas.shape)
                elif source == "BULGE" :
                    hlradii = 1. * np.ones(sigmas.shape)
            
            return self.fiber_acceptance_func[source](np.array([hlradii.ravel(),sigmas.ravel(),offsets.ravel()]).T).reshape(original_shape)
        
        