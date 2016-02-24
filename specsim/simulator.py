# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Top-level manager for spectroscopic simulation.

A simulator is usually initialized from a configuration, for example:

    >>> simulator = Simulator('test')
"""
from __future__ import print_function, division

import math

import numpy as np
import scipy.sparse as sp

from astropy import units as u

import specsim.config
import specsim.atmosphere
import specsim.instrument
import specsim.source


class QuickCamera(object):
    """
    A class representing one camera in a quick simulation.
    """
    def __init__(self, camera):
        wavelengthGrid = camera.wavelength.value
        self.sigmaWavelength = camera.rms_resolution.value
        self.throughput = camera.throughput
        self.readnoisePerBin = camera.read_noise_per_bin.value
        self.darkCurrentPerBin = camera.dark_current_per_bin.value
        self.gain = camera.gain.value
        self.coverage = camera.ccd_coverage

        # Truncate our throughput to the wavelength range covered by all fibers.
        self.throughput[~self.coverage] = 0.

        # extrapolate null values
        mask=np.where(self.sigmaWavelength<=0)[0]
        if mask.size > 0 and mask.size != wavelengthGrid.size:
            self.sigmaWavelength[mask] = np.interp(
                wavelengthGrid[mask], wavelengthGrid[self.sigmaWavelength>0],
                self.sigmaWavelength[self.sigmaWavelength>0])

        # Build a sparse matrix representation of the co-added resolution
        # smoothing kernel. Tabulate a Gaussian approximation of the PSF at each
        # simulation wavelength, truncated at 5x the maximum sigma.
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
        self.sparseKernel = sp.csc_matrix(
            (sparseData,sparseIndices,sparseIndPtr),(nbins,nbins)).tocsr()
        #self.sparseKernel = sp.csr_matrix(
        #   (sparseData,sparseIndices,sparseIndPtr),(nbins,nbins))


class Simulator(object):
    """
    Manage the simulation of an atmosphere, instrument, and source.

    Parameters
    ----------
    config : specsim.config.Configuration or str
        A configuration object or configuration name.
    """
    def __init__(self, config):

        if isinstance(config, basestring):
            config = specsim.config.load_config(config)

        self.atmosphere = specsim.atmosphere.initialize(config)
        self.instrument = specsim.instrument.initialize(config)
        self.source = specsim.source.initialize(config)
        self.downsampling = config.simulator.downsampling

        # Lookup the telescope's effective area in cm^2.
        self.effArea = self.instrument.effective_area.to(u.cm**2).value

        # Lookup the fiber area in arcsec^2.
        self.fiberArea = self.instrument.fiber_area.to(u.arcsec**2).value

        self.wavelengthGrid = config.wavelength.to(u.Angstrom).value

        self.cameras = [ ]
        for camera in self.instrument.cameras:
            quick_camera = QuickCamera(camera)
            self.cameras.append(quick_camera)


    def simulate(self):
        """Simulate a single exposure.

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
        downsampling = self.downsampling
        airmass = self.atmosphere.airmass
        expTime = self.instrument.exposure_time.to(u.s).value

        # Convert the photon response in our canonical flux units.
        photonRatePerBin = self.instrument.photons_per_bin.to(
            1e17 * u.Angstrom * u.cm**2 / u.erg).value

        # Convert the sky spectrum to our canonical units.
        sky = self.atmosphere.surface_brightness.to(
            1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom * u.arcsec**2)).value

        # Integrate the sky flux over the fiber area and convert to a total
        # photon rate (Hz).
        skyPhotonRate = sky * self.fiberArea * photonRatePerBin

        # Lookup the fiber acceptance function for this source type.
        self.fiberAcceptanceFraction = self.instrument.get_fiber_acceptance(
            self.source)

        # Convert the source spectrum flux to our canonical units.
        self.sourceFlux = self.source.flux_out.to(
            1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom)).value

        # Loop over cameras.
        sourcePhotonsSmooth = { }
        self.observedFlux = np.zeros_like(self.wavelengthGrid)
        throughputTotal = np.zeros_like(self.wavelengthGrid)
        for camera in self.cameras:

            # Rescale the mean rate (Hz) of photons to account for this camera's
            # throughput. We are including camera throughput but not
            # fiber acceptance losses or atmostpheric absorption here.
            camera.photonRatePerBin = camera.throughput * photonRatePerBin

            # Apply resolution smoothing and throughput to the sky photon rate.
            camera.skyPhotonRateSmooth = camera.sparseKernel.dot(
                camera.throughput * skyPhotonRate)

            # Calculate the calibration from source flux to mean detected photons
            # before resolution smearing in this camera's CCD.
            camera.sourceCalib = (expTime*camera.photonRatePerBin*
                self.fiberAcceptanceFraction * self.atmosphere.extinction)

            # Apply resolution smoothing to the mean detected photons response.
            camera.sourcePhotonsSmooth = camera.sparseKernel.dot(
                self.sourceFlux*camera.sourceCalib)

            # Truncate any resolution leakage beyond this camera's wavelength limits.
            camera.sourcePhotonsSmooth[~camera.coverage] = 0.

            # Calculate the variance in the number of detected electrons
            # for this camera (assuming one electron per photon).
            camera.nElecVariance = (
                camera.sourcePhotonsSmooth + camera.skyPhotonRateSmooth*expTime +
                (camera.readnoisePerBin)**2 + camera.darkCurrentPerBin*expTime)

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
            # Calculate readnoise squared.
            rdnoiseSq = np.sum(camera.readnoisePerBin[:last].reshape(downShape)**2,axis=1)
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
