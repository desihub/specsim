# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command-line script for simulating a fiber spectrograph.
"""
from __future__ import print_function, division

# script that runs galsim with all possible conditions
# and save fiber acceptance fraction as a function
# of various parameters in a single fits file
#
# the ouput file has to be copied to
# $DESIMODEL/data/throughput/galsim-fiber-acceptance.fits

import sys
import numpy as np
import astropy.table
import astropy.units as u
import astropy.io.fits as pyfits

import specsim.simulator
import specsim.fiberloss
import scipy.interpolate

def generate_fiber_positions(nfiber, seed, desi):
    """
    returns random fiber location on focal surface
    
    Args:
        nfibers (int) number of fiber
        seed (int) random seed
        desi specsim.simulator.Simulator object (need to init first with a config)
    Returns:
        x,y 2 1D np.array of size nfibers, random fiber location on focal surface
    """
    gen = np.random.RandomState(seed)
    focal_r = (
        np.sqrt(gen.uniform(size=nfiber)) * desi.instrument.field_radius)
    phi = 2 * np.pi * gen.uniform(size=nfiber)
    return np.cos(phi) * focal_r, np.sin(phi) * focal_r

def main(args=None):
    """
    fitgalsim runs the galsim fiber acceptance calculation
    using class specsim.fiberloss.GalsimFiberlossCalculator
    and saves the mean and rms acceptance as a function
    of source profile (point source 'POINT', exponential 'DISK',
    De Vaucouleurs 'BULGE'), effective PSF sigma (atmosphere+telescope blur),
    in um on focal surface, fiber offset from source on focal surface in um,
    and, for the extended source, the half light radius in arcsec.

    The method consists in first setting sigma and offset grid, and
    reverse engineering the atmospheric seeing to retrieve the correct
    effective sigma on the grid given the telescope blur.
    
    For each point in the output parameter grid (source type , sigma, offset,
    source radius), several calculations are done with random angular 
    orientation of fiber and source to account for the fiber ellipticity
    (due to anisotropic plate scale) and source ellipticity.

    The output file has to be saved in 
    $DESIMODEL/data/throughput/galsim-fiber-acceptance.fits to be used
    for fast fiber acceptance computation.
    This idea is to compute accurate and correlated values of offset, blur,
    scale, atmospheric seeing from the fiber location and wavelength, 
    compute the effective sigma and read with a ND interpolation the
    fiber acceptance value from the file.
    """
    # parameters
    ########################################################################"
    seed=0
    nsigma=11
    fwhm_to_sigma = 1./2.35482
    min_sigma=0.6*fwhm_to_sigma # arcsec
    max_sigma=3.*fwhm_to_sigma # arcsec
    noffset=30
    max_offset=2. # arcsec
    nrand=12 # randoms
    half_light_radii  = np.linspace(0.3,2.,20-3+1) # half light radius in arcsec
    axis_ratio = 0.7 # a fixed axis ratio is used for DISK and BULGE (position angles are random)
    sources = ["POINT","DISK","BULGE"]
    ########################################################################
    
    print("init simulator")
    
    desi    = specsim.simulator.Simulator('desi')
    wave    = np.linspace(6000.,6001.,nsigma) # Angstrom , wavelength are not used
    
    # optics with random fiber positions to get the range of scale and blur
    x,y = generate_fiber_positions(nrand, seed, desi)
    x=x.to(u.um).value
    y=y.to(u.um).value
    scale, blur, unused_offset = desi.instrument.get_focal_plane_optics(x*u.um, y*u.um, wave*u.angstrom)
    scale  = scale.to(u.um / u.arcsec).value
    blur   = blur.to(u.um).value
    mblur  = np.sqrt(np.mean(blur**2)) # quadratic mean
    mscale = np.sqrt(np.mean(scale[:,0]*scale[:,1])) # quadratic mean
    # I ignore the offsets from the random locations

    offset  = np.linspace(0,max_offset,noffset)*mscale # this is the final offset array I will save, um
    sigma   = np.linspace(min_sigma,max_sigma,nsigma)*mscale # this is the final sigma array I will save, um
    
    # random orientations of sources (account for source ellipticity)
    position_angle_source_deg = 360.*np.random.uniform(size=nrand)
    
    # random orientations of offsets (account for plate scale asymetry)
    theta = 2*np.pi*np.random.uniform(size=nrand)
    rcos_offset = np.cos(theta)
    rsin_offset = np.sin(theta)
    
    # init galsim calculator
    fiber_diameter = desi.instrument.fiber_diameter.to(u.um).value
    calc = specsim.fiberloss.GalsimFiberlossCalculator(fiber_diameter=fiber_diameter,wlen_grid=wave,
                                                       num_pixels=16,oversampling=32,moffat_beta=3.5)

    hdulist=None
    
    for source in sources :
        
        nfibers=noffset
        disk_bulge_fraction    = np.zeros((nfibers,2)) # fraction of disk and bulge
        minor_major_axis_ratio = axis_ratio*np.ones((nfibers,2)) # minor / major axis ratio , for disk and bulge component, respectively
        position_angle         = np.zeros((nfibers,2)) # deg

        source_half_light_radii = None

        if source=="POINT" :
            source_half_light_radii=[0] # none for point source
        elif source=="DISK" :
            disk_bulge_fraction[:,0]=1
            source_half_light_radii=half_light_radii
        elif source=="BULGE" :
            disk_bulge_fraction[:,1]=1
            source_half_light_radii=half_light_radii

        mean_loss=[]
        rms_loss=[]
        for half_light_radius_value in source_half_light_radii :

            half_light_radius      = half_light_radius_value * np.ones((nfibers,2))

            print("computing fiberloss for",source,"hlr=",half_light_radius_value,"arcsec")
            sys.stdout.flush()
            
            sloss=np.zeros((noffset,nsigma)) # sum of loss values
            sloss2=np.zeros((noffset,nsigma)) # sum of loss2 values
            for r in range(nrand) :
                blur2=np.mean(blur[r,:]**2) # scalar, mean blur
                scale2=scale[r,0]*scale[r,1] # scalar ,sigmax*sigmay
                
                # we artificially set the seeing array to span the seeing range instead of following
                # evolution with wavelength
                #
                # this is the inverse of (in fiberloss.py) :
                # sigma[i] = np.sqrt( (seeing_fwhm/2.35482)**2*scale_um_per_arcsec[i,0]*scale_um_per_arcsec[i,1]+blur_um[i]**2 )
                
                atmospheric_seeing = np.sqrt((sigma**2 - blur2)/scale2)/fwhm_to_sigma # size nsigma , arcsec, fwhm
                
                galsim_scale  = np.zeros((noffset,2))
                galsim_offset = np.zeros((noffset,nsigma,2))
                galsim_blur   = np.zeros((noffset,nsigma))
                galsim_scale[:,0] = scale[r,0] # use actual scale of random locations, radial term
                galsim_scale[:,1] = scale[r,1] # use actual scale of random locations, tangential term
                for i in range(noffset) :
                    galsim_blur[i,:]  = blur[r,:] # same blur as a function of wavelength
                    galsim_offset[i,:,0] = offset[i]*rcos_offset[r] # apply a random angle to the offset
                    galsim_offset[i,:,1] = offset[i]*rsin_offset[r] # apply a random angle to the offset
                position_angle[:,:]  = position_angle_source_deg[r] # degrees 
                                    
                # calculate using galsim (that's long)
                loss = calc.calculate(seeing_fwhm=atmospheric_seeing,scale=galsim_scale,
                                      offset=galsim_offset,blur_rms=galsim_blur,
                                      source_fraction=disk_bulge_fraction,source_half_light_radius=half_light_radius,
                                      source_minor_major_axis_ratio=minor_major_axis_ratio,
                                      source_position_angle=position_angle)
            
                sloss += loss
                sloss2 += loss**2
            mloss=sloss/nrand 
            mean_loss.append( mloss.T )
            rms_loss.append( np.sqrt(sloss2/nrand-mloss**2).T )

        if hdulist is None :
            hdulist=pyfits.HDUList([pyfits.PrimaryHDU(sigma),pyfits.ImageHDU(offset,name="OFFSET"),pyfits.ImageHDU(half_light_radii,name="HLRAD")])
            header=hdulist[0].header
            header["EXTNAME"]="SIGMA"
            # add stuff ...
            header.add_comment("HDU SIGMA  = square root of quadratic sum of atmospheric seeing and blur, in um on focal surface, (both in rms ~ fwhm/2.35)")
            header.add_comment("HDU OFFSET = fiber offset in um on focal surface")
            header.add_comment("HDU HLRAD  = half light radii in arcsec (for DISK and BULGE)")
            header.add_comment("HDU POINT  = 2D image of average fiber acceptance vs SIGMA and OFFSET")
            header.add_comment("HDU DISK   = 3D image of average fiber acceptance vs SIGMA, OFFSET, and HLRAD")
            header.add_comment("HDU BULGE  = 3D image of average fiber acceptance vs SIGMA, OFFSET, and HLRAD")
            header.add_comment("HDU PRMS   = 2D image of fiber acceptance rms for POINT source vs SIGMA and OFFSET")
            header.add_comment("HDU DRMS   = 3D image of fiber acceptance rms for DISK profile vs SIGMA, OFFSET, and HLRAD")
            header.add_comment("HDU BRMS   = 3D image of fiber acceptance rms for BULGE profile vs SIGMA, OFFSET, and HLRAD")
            header.add_comment("using specsim.fiberloss.GalsimFiberlossCalculator")
            header.add_comment("galsim DISK profile: exponential")
            header.add_comment("galsim BULGE profile: DeVaucouleurs")
            header["MSCALE"]=(mscale,"plate scale in um/arcsec (use for plots only)")
            header["NRAND"]=(nrand,"number of random measurements (to average position angles)")
            header["AXRATIO"]=(axis_ratio,"axis ratio for extended sources")
            moffat_beta=3.5
            header.add_comment("galsim Atmospheric PSF: Moffat with beta=%2.1f"%moffat_beta)
            header.add_comment("galsim Telescope blur PSF: Gaussian")
            

        if len(mean_loss)>1 :
            mean_loss=np.array(mean_loss)
            rms_loss=np.array(rms_loss)
        else :
            mean_loss=mean_loss[0]
            rms_loss=rms_loss[0]
        hdulist.append(pyfits.ImageHDU(mean_loss,name=source))
        hdulist.append(pyfits.ImageHDU(rms_loss,name=source[0]+"RMS"))

    ofilename="galsim-fiber-acceptance.fits"
    hdulist.writeto(ofilename,overwrite=True)
    print("wrote",ofilename)
