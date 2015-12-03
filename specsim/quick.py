# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provides Quick class for quick simulations of DESI observations.
"""
from __future__ import print_function, division

import math
import numpy as np
import scipy.sparse as sp
import scipy.interpolate
from scipy.special import exp10
from astropy import constants as const
from astropy import units

import specsim.spectrum
import specsim.atmosphere
import specsim.instrument


class QuickCamera(object):
    """
    A class representing one camera in a quick simulation.
    """
    def __init__(self,wavelengthRange,throughputModel,psfFWHMWavelengthModel,angstromsPerRowModel,
        psfNPixelsSpatialModel,readnoisePerPixel,darkCurrentPerPixel):
        self.wavelengthRange = wavelengthRange
        self.throughputModel = throughputModel
        self.psfFWHMWavelengthModel = psfFWHMWavelengthModel
        self.angstromsPerRowModel = angstromsPerRowModel
        self.psfNPixelsSpatialModel = psfNPixelsSpatialModel
        self.readnoisePerPixel = readnoisePerPixel
        self.darkCurrentPerPixel = darkCurrentPerPixel

    def setWavelengthGrid(self,wavelengthGrid,photonRatePerBin,skyPhotonRate):
        """
        setWavelengthGrid

        Initializes for the specified wavelength grid which is guaranteed to be
        linearly spaced. The mean rate of expected photons per wavelength bin
        corresponding to a unit flux (assuming 100%% throughput) is provided via
        photonRatePerBin. The mean sky photon rate per wavelength bin is
        provided via skyPhotonRate.
        """
        # Resample our throughput on this grid.
        self.throughput = self.throughputModel.getResampledValues(wavelengthGrid)

        # Truncate our throughput to the wavelength range covered by all fibers.
        self.coverage = np.logical_and(
            wavelengthGrid >= self.wavelengthRange[0],wavelengthGrid <= self.wavelengthRange[1])
        self.throughput[~self.coverage] = 0.
        assert np.all(self.throughput[self.coverage] > 0), "Camera has zero throughput within wavelength limits"

        # Resample the PSF FWHM to our wavelength grid and convert from
        # FWHM to an equivalent Gaussian sigma.
        fwhmToSigma = 1./(2*math.sqrt(2*math.log(2)))
        self.sigmaWavelength = self.psfFWHMWavelengthModel.getResampledValues(wavelengthGrid)*fwhmToSigma

        # extrapolate null values
        mask=np.where(self.sigmaWavelength<=0)[0]
        if mask.size > 0 and mask.size != wavelengthGrid.size :
            self.sigmaWavelength[mask]=np.interp(wavelengthGrid[mask],wavelengthGrid[self.sigmaWavelength>0],self.sigmaWavelength[self.sigmaWavelength>0])
        

        # Resample the conversion between pixels and Angstroms and save the results.
        self.angstromsPerRow = self.angstromsPerRowModel.getResampledValues(wavelengthGrid)

        # Build a sparse matrix representation of the co-added resolution smoothing kernel.
        # Tabulate a Gaussian approximation of the PSF at each simulation wavelength,
        # truncated at 5x the maximum sigma.
        max_sigma = np.max(self.sigmaWavelength)
        wavelength_spacing = wavelengthGrid[1] - wavelengthGrid[0]
        nhalf = int(np.ceil(5 * max_sigma / wavelength_spacing))
        nbins = wavelengthGrid.size
        sparseData = np.empty(((2 * nhalf + 1) * nbins,))
        sparseIndices = np.empty(((2 * nhalf + 1) * nbins,), dtype=np.int32)
        sparseIndPtr = np.empty((nbins + 1,), dtype=np.int32)
        nextIndex = 0

        # define a set of bins to compute the kernel
        # it is the range of positive throughput with a nhalf margin
        mask=np.where(self.throughput>0)[0]
        if mask.size > 0 :
            begin_bin = max(0,mask[0]-nhalf)
            end_bin   = min(nbins,mask[-1]+nhalf+1)
        else :
            begin_bin = nbins
            end_bin = nbins
            
        
        
        for bin in range(nbins):
            sparseIndPtr[bin] = nextIndex
            lam = wavelengthGrid[bin]
            sigma = self.sigmaWavelength[bin]
            if bin >= begin_bin and bin < end_bin and sigma > 0:
                first_row = max(0, bin - nhalf)
                last_row = min(nbins, bin + nhalf + 1)
                psf = np.exp(-0.5 * (
                    (wavelengthGrid[first_row: last_row] - lam) / sigma)**2)
                # We normalize the PSF even when it is truncated at the edges of the
                # resolution function, so that the resolution-convolved flux does not
                # drop off when the true flux is constant.
                # note: this preserves the flux integrated over the full PSF extent 
                # but not at smaller scales.
                psf /= np.sum(psf)
                rng = slice(nextIndex, nextIndex + psf.size)
                sparseIndices[rng] = range(first_row, last_row)
                sparseData[rng] = psf
                nextIndex += psf.size
        sparseIndPtr[-1] = nextIndex
        # The original IDL code uses the transpose of the correct smoothing kernel,
        # which corresponds to the second commented line below. We ultimately want the
        # kernel in CSR format since this is ~10% faster than CSC for the convolution.
        self.sparseKernel = sp.csc_matrix((sparseData,sparseIndices,sparseIndPtr),(nbins,nbins)).tocsr()
        #self.sparseKernel = sp.csr_matrix((sparseData,sparseIndices,sparseIndPtr),(nbins,nbins))

        '''
        # Build a non-sparse matrix as a cross check of the sparse version. This is only practical
        # for small wavelength grids and so is commented out by default.
        kernel = np.zeros((nbins,nbins))
        for bin in range(nhalf,nbins-nhalf):
            if self.throughput[bin] > 0:
                lam = wavelengthGrid[bin]
                sigma = self.sigmaWavelength[bin]
                if sigma > 0:
                    sup = slice(bin-nhalf,bin+nhalf)
                    psf = np.exp(-0.5*((wavelengthGrid[sup]-lam)/sigma)**2)
                    # I think [bin,sup] should be [sup,bin] here, but this matches the IDL
                    kernel[bin,sup] = psf/np.sum(psf)
        print('sparse ok?',np.array_equal(kernel,self.sparseKernel.T.todense()))
        '''

        # Rescale the mean rate (Hz) of photons to account for this camera's throughput.
        # We are including throughput but not atmostpheric absorption here.
        self.photonRatePerBin = self.throughput*photonRatePerBin

        # Apply resolution smoothing and throughput to the sky photon rate.
        self.skyPhotonRateSmooth = self.sparseKernel.dot(self.throughput*skyPhotonRate)

        # Resample the effective number of pixels in the spatial direction that
        # contribute read noise to each wavelength bin.
        nPixelsSpatial = self.psfNPixelsSpatialModel.getResampledValues(wavelengthGrid)

        # Calculate the pixel size of each wavelength bin.
        wavelengthStep = wavelengthGrid[1] - wavelengthGrid[0]
        nPixelsWavelength = np.zeros_like(wavelengthGrid)
        ccdMask = self.angstromsPerRow > 0
        nPixelsWavelength[ccdMask] = wavelengthStep/self.angstromsPerRow[ccdMask]

        # Calculate the effective number of pixels contributing to the signal in each wavelength bin.
        nEffectivePixels = nPixelsWavelength*nPixelsSpatial

        # Calculate the read noise in electrons per wavelength bin, assuming that
        # readnoise is uncorrelated between pixels (hence the sqrt scaling). The
        # value will be zero in pixels that are not used by this camera.
        self.readnoisePerBin = self.readnoisePerPixel*np.sqrt(nEffectivePixels)
        assert np.all(self.readnoisePerBin[~self.coverage]==0), "Readnoise nonzero outside coverage"

        # Calculate the dark current in electrons/s per wavelength bin.
        self.darkCurrentPerBin = self.darkCurrentPerPixel*nEffectivePixels

class Quick(object):
    """
    A class for quick simulations of DESI observations.

    Initializes a Quick simulation for the specified atmosphere and instrument.
    If either of these is None, they are initialized to their default state
    using the specified base path.
    """
    def __init__(self,atmosphere=None,instrument=None,basePath=''):
        self.atmosphere = (atmosphere if atmosphere else
            specsim.atmosphere.Atmosphere(basePath=basePath))
        self.instrument = (instrument if instrument else
            specsim.instrument.Instrument(basePath=basePath))

        # Precompute the physical constant h*c in units of erg*Ang.
        self.hc = const.h.to(units.erg*units.s)*const.c.to(units.angstrom/units.s)

        # Calculate the telescope's effective area in cm^2.
        dims = self.instrument.model['area']
        D = dims['M1_diameter'] # in meters
        obs = dims['obscuration_diameter'] # in meters
        trussArea = 0.5*(D - obs)*dims['M2_support_width'] # in meters^2
        self.effArea = 1e4*(math.pi*((0.5*D)**2 - (0.5*obs)**2) - 4*trussArea)

        # Calculate the fiber area in arcsec^2.
        fibers = self.instrument.model['fibers']
        self.fiberArea = math.pi*(0.5*fibers['diameter_arcsec'])**2

        # Loop over cameras in increasing wavelength order.
        self.cameras = [ ]
        for band,limits in zip(self.instrument.cameraBands,self.instrument.cameraWavelengthRanges):
            # Lookup this camera's RMS read noise in electrons/pixel.
            readnoisePerPixel = self.instrument.model['ccd'][band]['readnoise']
            # Lookup this camera's dark current in electrons/pixel/hour and convert /hr to /s.
            darkCurrentPerPixel = self.instrument.model['ccd'][band]['darkcurrent']/3600.
            # Initialize this camera.
            camera = QuickCamera(limits,self.instrument.throughput[band],
                self.instrument.psfFWHMWavelength[band],self.instrument.angstromsPerRow[band],
                self.instrument.psfNPixelsSpatial[band],readnoisePerPixel,darkCurrentPerPixel)
            self.cameras.append(camera)

        # No wavelength grid has been set yet.
        self.wavelengthGrid = None

    def setWavelengthGrid(self,wavelengthMin,wavelengthMax,wavelengthStep):
        """Set the linear wavelength grid to use for simulation.

        This method pre-tabulates simulation quantities on the specified grid that
        are independent of the simulated source, to avoid duplicating this work
        in subsequent repeated calls to :meth:`simulate`.

        In case the requested step size does not exactly divide the specified
        range, the maximum wavelength will be silently adjusted.

        The wavelength limits should normally be set ~5 sigma beyond the camera
        coverage to avoid artifacts from resolution fall off at the edges of
        the simulation grid.

        Parameters
        ----------
        wavelengthMin : float
            Minimum wavelength to simulate in Angstroms.
        wavelengthMax : float
            Maximum wavelength to simulate in Angstroms.
        wavelengthStep : float
            Linear step size to use in Angstroms.
        """
        nwave = 1+int(math.floor((wavelengthMax-wavelengthMin)/wavelengthStep))
        if nwave <= 0:
            raise ValueError('simulate.Quick: invalid wavelength grid parameters %r,%r,%r' %
                (wavelengthMin,wavelengthMax,wavelengthStep))
        wavelengthGrid = wavelengthMin + wavelengthStep*np.arange(nwave)

        # Are we already using this grid?
        if np.array_equal(wavelengthGrid,self.wavelengthGrid):
            return

        # Make sure we don't use an incompletely initialized grid.
        self.wavelengthGrid = None

        # Resample the fiberloss model for each source type.
        self.fiberAcceptanceFraction = { }
        for model in self.instrument.getSourceTypes():
            self.fiberAcceptanceFraction[model] = (
                self.instrument.fiberloss[model].getResampledValues(wavelengthGrid))

        # Calculate the energy per photon (ergs) at each wavelength.
        energyPerPhoton = self.hc.value/wavelengthGrid
        # Calculate the mean rate (Hz) of photons per wavelength bin for a flux of
        # 1e-17 erg/cm^2/s/Ang. We are assuming 100% throughput here.
        photonRatePerBin = 1e-17*self.effArea*wavelengthStep/energyPerPhoton

        # Resample the sky spectrum.
        sky = self.atmosphere.skySpectrum.getResampledValues(wavelengthGrid)

        # Integrate the sky flux over the fiber area and convert to a total photon rate (Hz).
        skyPhotonRate = sky*self.fiberArea*photonRatePerBin

        # Resample the zenith atmospheric extinction.
        self.extinction = self.atmosphere.zenithExtinction.getResampledValues(wavelengthGrid)

        # Initialize each camera for this wavelength grid.
        for camera in self.cameras:
            camera.setWavelengthGrid(wavelengthGrid,photonRatePerBin,skyPhotonRate)
            camera.sigma_wave=camera.sigmaWavelength
        # Remember this wavelength grid for subsequent calls to simulate().
        self.wavelengthGrid = wavelengthGrid

    def simulate(self,sourceType,sourceSpectrum,airmass=1.,nread=1.,expTime=None,downsampling=5):
        """
        simulate

        Simulates an observation of the specified source type and spectrum and at the specified
        air mass. The source type must be supported by the instrument model (as specified by
        Instrument.getSourceTypes). The source spectrum should either be a SpectralFluxDensity
        object, or else should contain equal-length arrays of wavelength in Angstroms
        and flux in units of 1e-17 erg/(cm^2*s*Ang) as sourceSpectrum[0:1].

        Uses the specified exposure time (secs) or the instrument's nominal exposure time if expTime
        is None. The read noise is scaled by sqrt(nread).

        Use the setWavelengthGrid() method to specify the grid of wavelengths used for
        this simulation, otherwise the default is to use the atmosphere's sky spectrum
        wavelength grid. After applying resolution smoothing, the results are downsampled in
        wavelength using the specified factor.

        Returns a numpy array of results with one row per downsampled wavelength bin containing
        the following named columns in a numpy.recarray:

         - wave: wavelength in Angstroms
         - srcflux: source flux in 1e-17 erg/s/cm^2/Ang
         - obsflux: estimate of observed co-added flux in 1e-17 erg/s/cm^2/Ang
         - ivar: inverse variance of obsflux in 1/(1e-17 erg/s/cm^2/Ang)**2
         - snrtot: total signal-to-noise ratio of coadded spectrum
         - nobj[j]: mean number of observed photons from the source in camera j
         - nsky[j]: mean number of observed photons from the sky in camera j
         - rdnoise[j]: RMS read noise in electrons in camera j
         - dknoise[j]: RMS dark current shot noise in electrons in camera j
         - snr[j]: signal-to-noise ratio in camera j
         - camivar[j]: inverse variance of source flux in (1e-17 erg/s/cm^2/Ang)^-2 in camera j
         - camflux[j]: source flux in 1e-17 erg/s/cm^2/Ang in camera j 
                     (different for each camera because of change of resolution)
                
         

        Note that the number of cameras (indexed by j) in the returned results is not hard coded
        but determined by the instrument we were initialized with. Cameras are indexed in order
        of increasing wavelength.

        After calling this method the following high-resolution (pre-downsampling) arrays are also
        available as data members:

         - wavelengthGrid ~ wave
         - sourceFlux ~ srcflux
         - observedFlux ~ obsflux

        In addition, each camera provides the following arrays tabulated on the same high-resolution
        wavelength grid:

         - throughput
         - sourcePhotonsSmooth ~ nobj
         - skyPhotonRateSmooth ~ nsky/expTime
         - readnoisePerBin ~ rdnoise
         - darkCurrentPerBin ~ sqrt(dknoise/expTime)

        These are accessible using the same camera index j, e.g. qsim.cameras[j].throughput.
        """
        # Check that this is a supported source type.
        if sourceType not in self.instrument.fiberloss:
            raise RuntimeError('Quick.simulate: source type %s is not one of %s.' %
                (sourceType,','.join(self.instrument.getSourceTypes())))

        # Use the instrument's nominal exposure time by default.
        if expTime is None:
            expTime = self.instrument.model['exptime']

        # Use a wavelength grid covering all cameras with 0.1 Ang spacing by default.
        if self.wavelengthGrid is None:
            waveMin = self.instrument.cameraWavelengthRanges[0][0]
            waveMax = self.instrument.cameraWavelengthRanges[-1][-1]
            self.setWavelengthGrid(waveMin,waveMax,0.1)

        # Convert the source to a SpectralFluxDensity if necessary, setting the flux to zero
        # outside the input source spectrum's range.
        if not isinstance(sourceSpectrum,specsim.spectrum.SpectralFluxDensity):
            sourceSpectrum = specsim.spectrum.SpectralFluxDensity(
                sourceSpectrum[0],sourceSpectrum[1],extrapolatedValue=0.)

        # Resample the source spectrum to our simulation grid, if necessary.
        self.sourceFlux = sourceSpectrum.values
        if not np.array_equal(self.wavelengthGrid,sourceSpectrum.wavelength):
            self.sourceFlux = sourceSpectrum.getResampledValues(self.wavelengthGrid)

        # Loop over cameras.
        sourcePhotonsSmooth = { }
        self.observedFlux = np.zeros_like(self.wavelengthGrid)
        throughputTotal = np.zeros_like(self.wavelengthGrid)
        for camera in self.cameras:

            # Calculate the calibration from source flux to mean number of detected photons
            # before resolution smearing in this camera's CCD.
            camera.sourceCalib = (expTime*camera.photonRatePerBin*
                self.fiberAcceptanceFraction[sourceType]*exp10(-self.extinction*airmass/2.5))

            # Apply resolution smoothing to the detected source photons.
            camera.sourcePhotonsSmooth = camera.sparseKernel.dot(
                self.sourceFlux*camera.sourceCalib)

            # Truncate any resolution leakage beyond this camera's wavelength limits.
            camera.sourcePhotonsSmooth[~camera.coverage] = 0.

            # Calculate the variance in the number of detected electrons
            # for this camera.
            camera.nElecVariance = (
                camera.sourcePhotonsSmooth + camera.skyPhotonRateSmooth*expTime +
                (camera.readnoisePerBin*nread)**2 + camera.darkCurrentPerBin*expTime)

            # Estimate this camera's contribution to the coadded observed source flux.
            self.observedFlux += camera.throughput*camera.sparseKernel.dot(self.sourceFlux)
            throughputTotal += camera.throughput

        # Approximate the observed source flux spectrum as the throughput-weighted sum of
        # the flux smoothed by each camera's resolution.
        thruMask = throughputTotal > 0
        self.observedFlux[thruMask] /= throughputTotal[thruMask]

        # Prepare the downsampled grid.
        nbins = self.wavelengthGrid.size
        try:
            ndown = nbins//downsampling
            assert ndown > 0
            # Truncate at the end of the spectrum in case the downsampling does not evenly divide
            # the high-resolution wavelength spectrum.
            last = ndown*downsampling
            downShape = (ndown,downsampling)
        except (TypeError,AssertionError):
            raise RuntimeError('simulate.Quick: invalid option downsampling = %r.' % downsampling)

        # Initialize the results record array. The str() wrappers below are required since we
        # are importing unicode_literals from __future__ but numpy does not accept unicode
        # record names.
        nbands = len(self.cameras)
        results = np.recarray((ndown,),dtype=[
            (str('wave'),float),
            (str('srcflux'),float),
            (str('obsflux'),float),
            (str('ivar'),float),
            #(str('ivarnew'),float),
            (str('snrtot'),float),
            (str('nobj'),float,(nbands,)),
            (str('nsky'),float,(nbands,)),
            (str('rdnoise'),float,(nbands,)),
            (str('dknoise'),float,(nbands,)),
            (str('snr'),float,(nbands,)),
            (str('camflux'),float,(nbands,)),
            (str('camivar'),float,(nbands,)),            
            ])

        # Fill the results arrays from our high-resolution arrays. Wavelengths are tabulated at
        # bin centers, so the first downsampled wavelength is offset by 0.5*(downsampling-1)*dwave
        # from the first high-resolution wavelength.
        dwave = (self.wavelengthGrid[-1]-self.wavelengthGrid[0])/(self.wavelengthGrid.size-1)
        results.wave = self.wavelengthGrid[:last:downsampling] + 0.5*(downsampling-1)*dwave
        results.srcflux = np.mean(self.sourceFlux[:last].reshape(downShape),axis=1)
        results.obsflux = np.mean(self.observedFlux[:last].reshape(downShape),axis=1)
        for j,camera in enumerate(self.cameras):
            # Add the source and sky photons contributing to each downsampled wavelength bin.
            (results.nobj)[:,j] = np.sum(camera.sourcePhotonsSmooth[:last].reshape(downShape),axis=1)
            # Scale the sky photon rate by the exposure time.
            (results.nsky)[:,j] = np.sum(camera.skyPhotonRateSmooth[:last].reshape(downShape),axis=1)*expTime
            # Calculate readnoise squared, scaled by the optional nread parameter (nominally 1).
            rdnoiseSq = np.sum(camera.readnoisePerBin[:last].reshape(downShape)**2,axis=1)*nread
            # Calculate the dark current shot noise variance.
            dknoiseSq = np.sum(camera.darkCurrentPerBin[:last].reshape(downShape),axis=1)*expTime
            # Save the noise contributions to the results.
            (results.rdnoise)[:,j] = np.sqrt(rdnoiseSq)
            (results.dknoise)[:,j] = np.sqrt(dknoiseSq)
            # Calculate the total variance in number of detected electrons of this downsampled bin.
            variance = (results.nobj)[:,j] + (results.nsky)[:,j] + rdnoiseSq + dknoiseSq
            # Calculate the corresponding signal-to-noise ratios. Bins with no signal will have SNR=0.
            signalMask = (results.nobj)[:,j] > 0
            (results.snr)[:,j] = np.zeros((ndown,))
            (results.snr)[signalMask,j] = (results.nobj)[signalMask,j]/np.sqrt(variance[signalMask])
            # Compute calib in downsampled wave grid, it's a sum because
            # nphot is a sum over orginal wave bins.
            # The effective calibration of convolved spectra is different from the true one
            # because resolution and transmission don't commute
            smooth_camera_calib=camera.sparseKernel.dot(camera.sourceCalib)
            calib_downsampled = np.sum(smooth_camera_calib[:last].reshape(downShape),axis=1)
            # Add inverse variance for camera
            vcMask=(variance>0)&(calib_downsampled>0)
            (results.camivar)[vcMask,j] = calib_downsampled[vcMask]**2/variance[vcMask]
            # Add flux in camera (not the same for all cameras because of change of resolution)
            (results.camflux)[vcMask,j] = (results.nobj)[vcMask,j]/calib_downsampled[vcMask]
            
        # Calculate the total SNR, combining the individual camera SNRs in quadrature.
        results.snrtot = np.sqrt(np.sum(results.snr**2,axis=1))
        # Calculate the corresponding inverse variance per bin. Bins with no observed flux will have IVAR=0.
        fluxMask = results.obsflux > 0
        results.ivar = np.zeros((ndown,))
        results.ivar[fluxMask] = (results[fluxMask].snrtot/results[fluxMask].obsflux)**2

        '''
        # Loop over downsampled bins to calculate flux inverse variances.
        results.ivarnew = np.zeros((ndown,))
        for alpha in range(ndown):
            # Initialize the weight vector for flux interpolation parameter alpha.
            wgt = np.zeros((nbins,))
            wgt[alpha*downsampling:(alpha+1)*downsampling] = 1.
            # Loop over cameras.
            for camera in self.cameras:
                # Calculate this camera's photons-per-unit-flux response to this parameter.
                Kprime = camera.sparseKernel.dot(camera.sourceCalib*wgt)
                # Calculate this camera's contribution to the corresponding diagonal
                # flux inverse variances.
                nIVar = np.zeros((nbins,))
                nIVar[camera.coverage] = 1./camera.nElecVariance[camera.coverage]
                fIVar = Kprime**2*(nIVar + 0.5*nIVar**2)
                results.ivarnew[alpha] += np.sum(fIVar[:last])
        np.savetxt('ivar.dat',np.vstack([results.ivar,results.ivarnew]).T)
        '''

        # Remember the parameters used for this simulation.
        self.airmass = airmass
        self.sourceType = sourceType
        self.expTime = expTime

        # Return the downsampled vectors
        return results

    def plot(self,results,labels=None,plotMin=None,plotMax=None,plotName='quicksim'):
        """
        plot

        Generates a pair of plots for the specified results from a previous call
        to simulate(...). Specify plotMin,Max in Angstroms to restrict the range
        of the plot. The caller is responsible for calling plt.show() and/or
        plt.savefig(...) after this method returns. The optional
        labels[0],labels[1] are used to label the two plots.
        """
        # Defer this import until we are actually asked to make a plot.
        import matplotlib.pyplot as plt
        # Select the colors used for each camera band. If there are more bands than colors, we
        # will cycle through them.
        colors = ((0,0,1),(1,0,0),(1,0,1))
        # Determine the wavelength range to plot.
        wave = results.wave
        waveMin,waveMax = wave[0],wave[-1]
        if plotMin and plotMin < waveMax:
            waveMin = max(waveMin,plotMin)
        if plotMax and plotMax > waveMin:
            waveMax = min(waveMax,plotMax)
        # Create an empty frame with the specified plot name.
        fig = plt.figure(plotName,figsize=(10,8))
        # Upper plot is in flux units.
        plt.subplot(2,1,1)
        plt.xlabel('Wavelength (Ang)')
        plt.ylabel('Flux, Flux Error / (1e-17 erg/s/cm^2/Ang)')
        plt.xlim((waveMin,waveMax))
        # Plot the source spectrum as a black curve.
        plt.plot(wave,results.srcflux,'k-')
        plt.fill_between(wave,results.srcflux,0.,color=(0.7,0.7,0.7))
        # Use the maximum source flux to set the vertical limits.
        ymax = 2*np.max(results.srcflux)
        plt.ylim((0.,ymax))
        # Superimpose the contributions to the flux error from each camera.
        elecNoise = np.sqrt(results.rdnoise**2 + results.dknoise**2)
        for j,camera in enumerate(self.cameras):
            snr = (results.snr)[:,j]
            mask = snr > 0
            color = colors[j%len(colors)]
            # Calculate and plot the total flux error in each camera.
            sigma = np.zeros_like(snr)
            sigma[mask] = results[mask].obsflux/snr[mask]
            plt.plot(wave[mask],sigma[mask],'.',markersize=2.,color=color,alpha=0.5)
            # Calculate and plot the flux error contribution due to read noise only.
            fluxNoise = np.zeros_like(snr)
            fluxNoise[mask] = results[mask].obsflux*(elecNoise)[mask,j]/(results.nobj)[mask,j]
            plt.plot(wave[mask],fluxNoise[mask],'.',markersize=2.,color=color,alpha=0.1)
        # Inset a label if requested.
        if labels:
            xpos,ypos = 0.05,0.9
            plt.annotate(labels[0],xy=(xpos,ypos),xytext=(xpos,ypos),
                xycoords='axes fraction',textcoords='axes fraction')
        # Lower plot is in SNR units.
        plt.subplot(2,1,2)
        plt.xlabel('Wavelength (Ang)')
        binSize = results.wave[1]-results.wave[0]
        plt.ylabel('Signal-to-Noise Ratio / (%.1f Ang)' % binSize)
        plt.xlim((waveMin,waveMax))
        # Superimpose the SNR contributions from each camera with transparent fills.
        for j,camera in enumerate(self.cameras):
            snr = (results.snr)[:,j]
            mask = snr > 0
            plt.fill_between(wave[mask],snr[mask],0.,alpha=0.3,color=colors[j%len(colors)])
        # Plot the total SNR with black points.
        plt.plot(wave,results.snrtot,'k.',markersize=2.)
        # Inset a label if requested.
        if labels:
            xpos,ypos = 0.05,0.9
            plt.annotate(labels[1],xy=(xpos,ypos),xytext=(xpos,ypos),
                xycoords='axes fraction',textcoords='axes fraction')
        # Clean up the plot aesthetics a bit.
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(left=0.06,bottom=0.06,right=0.97,top=0.98,hspace=0.14)
