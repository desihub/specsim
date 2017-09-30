#!/usr/bin/env python

# script that runs galsim with all possible conditions
# and save fiber acceptance fraction as a function
# of various parameters in a single fits file
#
# the ouput file has to be copied to $DESIMODEL/data/throughput/galsim-fiber-acceptance.fits

import sys
import numpy as np
import astropy.table
import astropy.units as u
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import time

import specsim.simulator
import specsim.fiberloss
from scipy.interpolate import RegularGridInterpolator

def prof2d(x,y,z,nx,ny) :
    x0=np.min(x)
    x1=np.max(x)
    y0=np.min(y)
    y1=np.max(y)
    xi=( nx*(x-x0)/(x1-x0) ).astype(int)
    yi=( ny*(y-y0)/(y1-y0) ).astype(int)
    ii=xi*ny+yi    
    bins=np.arange(nx*ny+1)-0.5
    h1,junk=np.histogram(ii,bins=bins)
    hx,junk=np.histogram(ii,bins=bins,weights=x)
    hy,junk=np.histogram(ii,bins=bins,weights=y)
    hz,junk=np.histogram(ii,bins=bins,weights=z)
    h1=h1.astype(float)
    hx/=(h1+(h1==0))
    hy/=(h1+(h1==0))
    hz/=(h1+(h1==0))
    
    # fill empty bins with linear fit
    bx1d=(np.arange(nx)+0.5)*((x1-x0)/nx)+x0
    by1d=(np.arange(ny)+0.5)*((y1-y0)/ny)+y0
    bx=(np.tile(bx1d,(ny,1)).T)
    by=(np.tile(by1d,(nx,1)))
    bx=bx.ravel()
    by=by.ravel()
    
    A=np.zeros((3,3))
    ok=(h1>0)
    A[0,0] = np.sum(ok)
    A[1,0] = A[0,1] = np.sum(hx[ok])
    A[1,1] = np.sum(hx[ok]**2)
    A[2,0] = A[0,2] = np.sum(hy[ok])
    A[2,1] = A[1,2] = np.sum(hx[ok]*hy[ok])
    A[2,2] = np.sum(hy[ok]**2)
    B=np.zeros((3))
    B[0] = np.sum(hz[ok])
    B[1] = np.sum(hz[ok]*hx[ok])
    B[2] = np.sum(hz[ok]*hy[ok])
    Ai = np.linalg.inv(A)
    C  = Ai.dot(B)
    bad=(h1==0)
    hz[bad]=C[0]+C[1]*bx[bad]+C[2]*by[bad]
    
    hx[bad]=bx[bad]
    hy[bad]=by[bad]
    h1[bad] += 0.0001 # so we can do averaging anyway
    h1=h1.reshape(nx,ny)
    hx=hx.reshape(nx,ny)
    hy=hy.reshape(nx,ny)
    hz=hz.reshape(nx,ny)
    x1d=np.sum(h1*hx,axis=1)/np.sum(h1,axis=1)
    y1d=np.sum(h1*hy,axis=0)/np.sum(h1,axis=0)
    
    return x1d,y1d,hz

def func2d(x1d,y1d,z2d) :
    return RegularGridInterpolator(points=(x1d,y1d),values=z2d,method="linear",bounds_error=False,fill_value=None)

def generate_fiber_positions(nfiber, seed, desi):
    gen = np.random.RandomState(seed)
    focal_r = (
        np.sqrt(gen.uniform(size=nfiber)) * desi.instrument.field_radius)
    phi = 2 * np.pi * gen.uniform(size=nfiber)
    return np.cos(phi) * focal_r, np.sin(phi) * focal_r

def generate_sources(nsrc, disk_fraction, bulge_fraction, seed , vary=""):
    gen = np.random.RandomState(seed)
    varied = vary.split(',')
    source_fraction = np.tile([disk_fraction, bulge_fraction], (nsrc, 1))
    source_half_light_radius = np.tile([0.45, 1.0], (nsrc, 1))
    source_minor_major_axis_ratio = np.tile([1.0, 1.0], (nsrc, 1))
    if 'pa' in varied:
        source_position_angle = 360. * gen.uniform(size=(nsrc, 2))
    else:
        source_position_angle = np.tile([30., 30.], (nsrc, 1))
    return source_fraction, source_half_light_radius, source_minor_major_axis_ratio, source_position_angle


# parameters
########################################################################"
seed=0
desi    = specsim.simulator.Simulator('desi') # the only one in package anyway ...
wave    = np.linspace(3550.,10000,8) # Angstrom
sources = ["POINT","DISK","BULGE"]
nfibers = 10000 # random positions in focal plane (and possibly random source orientation)
half_light_radii  = np.linspace(0.3,1.5,5) # half light radius in arcsec for disk=exponential and bulge=devaucouleurs profiles
total_seeing_fwhm = np.linspace(0.5,3.,5)  # FWHM in arcsec atmosphere + Mayall blur
########################################################################




# optics
R=desi.instrument.field_radius.to(u.um).value
x,y = generate_fiber_positions(nfibers, seed, desi)
x=x.to(u.um).value
y=y.to(u.um).value

scale, blur, offset = desi.instrument.get_focal_plane_optics(x*u.um, y*u.um, wave*u.angstrom)
scale  = scale.to(u.um / u.arcsec).value
offset = offset.to(u.um).value
blur   = blur.to(u.um).value

# offset from fiber to source
d2=offset[:,:,0]**2+offset[:,:,1]**2
d2=d2.ravel()

mscale=np.mean(np.sqrt(scale[:,0]*scale[:,1]))
print("mean scale  =",mscale,"um/arcsec")

# init galsim calculator
fiber_diameter = desi.instrument.fiber_diameter.to(u.um).value
calc = specsim.fiberloss.GalsimFiberlossCalculator(fiber_diameter=fiber_diameter,wlen_grid=wave,
                                                   num_pixels=16,oversampling=32,moffat_beta=3.5)


hdulist=None

for source in sources :
    
    
    disk_bulge_fraction    = np.zeros((nfibers,2)) # fraction of disk and bulge
    minor_major_axis_ratio = 1.*np.ones((nfibers,2)) # minor / major axis ratio , for disk and bulge component, respectively
    position_angle         = 1.*np.zeros((nfibers,2)) # not used because ellipcity=0

    source_half_light_radii = None
    
    if source=="POINT" :
        source_half_light_radii=[0] # none for point source
    elif source=="DISK" :
        disk_bulge_fraction[:,0]=1
        source_half_light_radii=half_light_radii
    elif source=="BULGE" :
        disk_bulge_fraction[:,1]=1
        source_half_light_radii=half_light_radii
    
    zz=[]
    zzrms=[]
    for half_light_radius_value in source_half_light_radii :

        half_light_radius      = half_light_radius_value * np.ones((nfibers,2))
        
        dd2=np.array([]) # array of values of offsets**2 
        ss2=np.array([]) # array of values of sigma**2
        ll=np.array([])  # array of values of fiber loss
    
        for seeing in total_seeing_fwhm :

            print("computing fiberloss for",source,"hlr=",half_light_radius_value,"arcsec, seeing=",seeing,"arcsec")
            sys.stdout.flush()
            
            desi.atmosphere._seeing['fwhm_ref'] = ( 2.35482 * np.sqrt((seeing/2.35482) ** 2 - 0.219**2) )
            atmospheric_seeing = desi.atmosphere.get_seeing_fwhm(wave*u.angstrom) # adds wavelength dependence

            # compute sky+telescope sigma2
            s2=np.zeros((offset.shape[0],offset.shape[1]))
            for i in range(offset.shape[0]) :
                s2[i] = ( (atmospheric_seeing/2.35482)**2*scale[i,0]*scale[i,1]+blur[i]**2 )
            ss2=np.append(ss2,s2.ravel())
            dd2=np.append(dd2,d2.ravel())
            # calculate using galsim (that's long)
            loss = calc.calculate(seeing_fwhm=atmospheric_seeing,scale=scale,offset=offset,blur_rms=blur,
                                  source_fraction=disk_bulge_fraction,source_half_light_radius=half_light_radius,
                                  source_minor_major_axis_ratio=minor_major_axis_ratio,
                                  source_position_angle=position_angle)
            ll=np.append(ll,loss.ravel())
        
        # average the fiberloss as a function of s2 and d2
        x,y,z=prof2d(ss2,dd2,ll,20,15)
        x,y,z2=prof2d(ss2,dd2,ll**2,20,15)
        var=z2-z**2
        zrms=np.sqrt(var*(var>0))
        zz.append(z)
        zzrms.append(zrms)
    
    
    if hdulist is None :
        hdulist=pyfits.HDUList([pyfits.PrimaryHDU(np.sqrt(x)),pyfits.ImageHDU(np.sqrt(y),name="OFFSET"),pyfits.ImageHDU(half_light_radii,name="HLRAD")])
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
        header["NRAND"]=(nfibers,"number of random fiber locations")
        moffat_beta=3.5
        header.add_comment("galsim Atmospheric PSF: Moffat with beta=%2.1f"%moffat_beta)
        header.add_comment("galsim Telescope blur PSF: Gaussian")        
        header.add_comment("computed with seeing from %3.2f to %3.2f arcsec FWHM"%(total_seeing_fwhm[0],total_seeing_fwhm[-1]))
        header.add_comment("computed with wavelength from %dA to %dA"%(wave[0],wave[-1]))
        
        hdulist.append(pyfits.ImageHDU(z,name="POINT"))
        hdulist.append(pyfits.ImageHDU(zrms,name="PRMS"))
    
    if len(zz)>1 :
        zz=np.array(zz)
        zzrms=np.array(zzrms)
    else :
        zz=zz[0]
        zzrms=zzrms[0]
    hdulist.append(pyfits.ImageHDU(zz,name=source))
    hdulist.append(pyfits.ImageHDU(zzrms,name=source[0]+"RMS"))

ofilename="galsim-fiber-acceptance.fits"
hdulist.writeto(ofilename,overwrite=True)
print("wrote",ofilename)


