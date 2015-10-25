# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provides classes to represent functions of wavelength.
"""
from __future__ import print_function, division

import math
import os.path
import numpy as np
import scipy.interpolate
import scipy.integrate
from astropy import constants as const
from astropy import units


class WavelengthFunction(object):
    """
    Represents an arbitrary function of wavelength.
    """
    def __init__(self,wavelength,values,wavelengthUnits=units.angstrom,valueUnits=None,
        extrapolatedValue=None):
        """
        Initializes a function of wavelength using tabulated values at specified wavelengths.
        The default wavelength units are angstroms but other units can be specified. Optional
        value units can also be specified but are not required. The input wavelengths must
        be increasing. The input wavelength and value arrays must have the same size. If either
        is already a numpy array, no internal copy is made (except when conversion to Angstroms
        is needed) so these are lightweight objects but be aware of possible side effects.
        """
        # Check that the input arrays have the same size.
        if len(wavelength) != len(values):
            raise RuntimeError('WavelengthFunction: wavelength and values arrays have different sizes.')
        # Save the input wavelength array as a numpy array in angstroms. If the input array is
        # already a numpy array in Angstroms, save a new view instead of copying it.
        convert = units.Unit(wavelengthUnits).to(units.angstrom)
        if isinstance(wavelength,np.ndarray) and convert == 1:
            self.wavelength = wavelength.view()
        else:
            self.wavelength = convert*np.array(wavelength)
        # Check for a strictly increasing wavelength grid.
        if not np.all(self.wavelength[1:] > self.wavelength[:-1]):
            raise RuntimeError('WavelengthFunction: input wavelengths are not strictly increasing.')
        # Save the input values as a numpy array or a new view into an existing numpy array.
        if isinstance(values,np.ndarray):
            self.values = values.view()
        else:
            self.values = np.array(values)
        # Remember the value units.
        self.valueUnits = valueUnits
        # Defer creating an interpolation model until we actually need one.
        self.extrapolatedValue = extrapolatedValue
        self.model = None

    def getModel(self):
        """
        Returns a model for interpolating within our tabulated wavelength function values.
        If an extrapolatedValue was provided to the constructor, the model will use this
        for any wavelengths outside the tabulated grid.
        """
        if self.model is None:
            self.model = scipy.interpolate.interp1d(self.wavelength,self.values,
                kind='linear',copy=False,bounds_error=(self.extrapolatedValue is None),
                fill_value=self.extrapolatedValue)
        return self.model

    def getResampledValues(self,wavelength,wavelengthUnits=units.angstrom):
        """
        Returns a numpy array of values resampled at the specified wavelengths (which do not
        need to be sorted). The default wavelength units are angstroms but other units can
        be specified. Raises a RuntimeError if resampling would require an extrapolation but
        no extrapolatedValue was provided to the constructor.
        """
        # Convert wavelengths to a numpy array in angstroms, if necessary.
        convert = units.Unit(wavelengthUnits).to(units.angstrom)
        if not isinstance(wavelength,np.ndarray) or convert != 1:
            wavelength = convert*np.array(wavelength)
        # Is the resampling array actually different from our input array?
        if np.array_equal(self.wavelength,wavelength):
            return self.values
        # Check that we have an extrapolatedValue if we need it.
        if self.extrapolatedValue is None and (np.min(wavelength) < self.wavelength[0] or
            np.max(wavelength) > self.wavelength[-1]):
            raise RuntimeError('WavelengthFunction: missing extrapolatedValue for resampling.')
        # Perform the resampling
        return self.getModel()(wavelength)

    def __call__(self,wavelength):
        """
        Returns the function value associated with the specified wavelength in Angstroms.
        """
        return self.getModel()(wavelength)

    def saveToTextFile(self,filename):
        """
        Writes a text file containing two columns: wavelength and values.
        """
        np.savetxt(filename,np.vstack([self.wavelength,self.values]).T)

    @classmethod
    def loadFromTextFile(cls,filename,wavelengthColumn=0,valuesColumn=1,
        wavelengthUnits=units.angstrom,extrapolatedValue=None):
        """
        Returns a new WavelengthFunction (or subclass of WavelengthFunction) from the specified
        text file.  Any comment lines beginning with '#' are ignored. Uses the specified columns
        for the wavelength and values. Additional columns are allowed and silently ignored.
        The default wavelength units are Angstroms but other units can be specified. Refer to
        the WavelengthFunction constructor for details on extrapolatedValue.
        """
        content = np.loadtxt(filename,unpack=True)
        if max(wavelengthColumn,valuesColumn) >= len(content):
            raise RuntimeError('WavelengthFunction: invalid columns for loadFromTextFile.')
        return cls(content[wavelengthColumn],content[valuesColumn],
            wavelengthUnits=wavelengthUnits,extrapolatedValue=extrapolatedValue)

class SpectralFluxDensity(WavelengthFunction):
    """
    Represents a spectral flux density as a function of wavelength.
    """
    def __init__(self,wavelength,flux,wavelengthUnits=units.angstrom,fluxUnits=None,
        extrapolatedValue=None):
        """
        Initializes a spectral flux density using tabulated flux values at specified wavelengths.
        See the documentation of WavelengthFunction for details. The default flux unit is
        1e-17 erg/(s*cm^2*Ang), which is available as SpectralFluxDensity.fiducialFluxUnit, but
        other units can be specified.
        """
        # Convert flux to a numpy array/view in our fiducial units.
        if fluxUnits is not None:
            convert = units.Unit(fluxUnits).to(self.fiducialFluxUnit)
        else:
            convert = 1
        if not isinstance(flux,np.ndarray) or convert != 1:
            flux = convert*np.array(flux)
        # Initialize our base WavelengthFunction.
        WavelengthFunction.__init__(self,wavelength,flux,wavelengthUnits,valueUnits=self.fiducialFluxUnit,
            extrapolatedValue=extrapolatedValue)

    def createRescaled(self,sdssBand,abMagnitude):
        """
        Returns a new SpectralFluxDensity whose flux has been rescaled to the specified AB magnitude
        in the specified SDSS band (identified by a character 'u','g','r','i','z'). Raises a
        RuntimeError in case our spectrum does not fully cover the band.
        """
        if sdssBand not in 'ugriz':
            raise RuntimeError('SpectralFluxDensity: invalid sdssBand %r' % sdssBand)
        # Calculate our magnitudes before rescaling.
        mags = self.getABMagnitudes()
        # Check that we fully cover this band.
        if mags[sdssBand] is None:
            raise RuntimeError('SpectralFluxDensity: %s-band is not fully covered.' % sdssBand)
        # Return a new spectrum with rescaled flux.
        scale = 10.**((mags[sdssBand]-abMagnitude)/2.5)
        extrap = self.extrapolatedValue
        if extrap is not None:
            extrap *= scale
        return SpectralFluxDensity(self.wavelength,scale*self.values,extrapolatedValue=extrap)

    def createRedshifted(self,newZ,oldZ=0.,preserveWavelengths=False):
        """
        Returns a new SpectralFluxDensity whose wavelengths and fluxes have been rescaled for the
        transformation from oldZ to newZ. If preserveWavelengths is True and an extrapolatedValue
        has been set, then the rescaled spectrum will be resampled to the original wavelengths.
        Otherwise, the new spectrum will be tabulated on the redshifted grid.
        """
        scale = (1.+newZ)/(1.+oldZ)
        newWave = self.wavelength*scale
        newFlux = self.values/scale
        extrap = self.extrapolatedValue
        if extrap is not None:
            extrap /= scale
        if preserveWavelengths:
            if extrap is None:
                raise RuntimeError('SpectralFluxDensity: need extrapolatedValue to redshift.')
            newFlux = self.getResampledValues(self.wavelength/scale)/scale
            return SpectralFluxDensity(self.wavelength,newFlux,extrapolatedValue=extrap)
        else:
            newFlux = self.values/scale
            return SpectralFluxDensity(self.wavelength*scale,newFlux,extrapolatedValue=extrap)

    def getFilteredRates(self,filterCurves,wavelengthStep=1.0):
        """
        Returns the counting rates in photons/(s*cm^2) when our spectral flux density is filtered by
        the specified curves. The curves should be specified as a dictionary of WavelengthFunctions
        and the results will also be a dictionary of floats using the same keys. Rates of None
        are returned when our spectrum has no extrapolatedValue set and a filter extends beyond
        our tabulated wavelengths.
        """
        rates = { }
        # Calculate the constant hc in erg*Ang.
        hc = (const.h.to(units.erg*units.s)*const.c.to(units.angstrom/units.s)).value
        # Loop over curves.
        for band,curve in filterCurves.iteritems():
            # Lookup this filter's wavelength limits.
            wmin,wmax = curve.wavelength[0],curve.wavelength[-1]
            # Do we cover this whole range?
            if self.extrapolatedValue is None and (wmin < self.wavelength[0] or wmax > self.wavelength[-1]):
                rates[band] = None
                continue
            # Build a wavelength grid covering these limits with a step size <= wavelengthStep.
            nwave = 1+int(math.ceil((wmax-wmin)/wavelengthStep))
            wave = np.linspace(wmin,wmax,nwave)
            # Resample the filter curve and our flux density to this wavelength grid.
            resampledCurve = curve.getResampledValues(wave)
            resampledFlux = self.getResampledValues(wave)
            # Estimate the integral using the trapezoid rule.
            integrand = resampledCurve*resampledFlux*wave/hc
            rates[band] = (np.sum(integrand[1:-1]) - 0.5*(integrand[0]+integrand[-1]))*(wmax-wmin)/nwave
        return rates

    def getABMagnitudes(self):
        """
        Returns a dictionary of AB magnitudes calculated in each SDSS filter. Magnitude values of
        None are returned when our spectrum has no extrapolatedValue set and a filter extends beyond
        our tabulated wavelengths.
        """
        if self.sdssFilterCurves is None:
            # Perform one-time initialization of sdssFilterCurves and sdssFilterRates.
            self.sdssFilterCurves = loadSDSSFilterCurves()
            # Tabulate the AB reference spectrum in units of 1e-17 erg/(s*cm^2*Ang) on a ~1 Ang grid.
            wmin = self.sdssFilterCurves['u'].wavelength[0]
            wmax = self.sdssFilterCurves['z'].wavelength[-1]
            nwave = 1+int(math.ceil((wmax-wmin)/1.))
            wave = np.linspace(wmin,wmax,nwave)
            abConst = (3631*units.Jy*const.c).to(units.erg/(units.cm**2*units.s)*units.angstrom).value
            flux = 1e17*abConst/wave**2
            abSpectrum = SpectralFluxDensity(wave,flux)
            # Calculate and save the AB reference counting rates in each band.
            self.sdssFilterRates = abSpectrum.getFilteredRates(self.sdssFilterCurves)
        assert self.sdssFilterRates is not None
        # Calculate the counting rates for our spectrum through each SDSS filter curve.
        rates = self.getFilteredRates(self.sdssFilterCurves)
        # Convert the rate ratios to AB magnitudes for all SDSS bands that we cover.
        mags = { }
        for band,rate in rates.iteritems():
            try:
                mags[band] = -2.5*math.log10(rate/self.sdssFilterRates[band])
            except (TypeError,ValueError):
                mags[band] = None
        return mags

    # Define our fiducial flux units.
    fiducialFluxUnit = 1e-17*units.erg/(units.s*units.cm**2*units.angstrom)

    # Placeholder for SDSS filter curves and counting rates used to calculate AB magnitudes.
    sdssFilterCurves = None
    sdssFilterRates = None

def loadSDSSFilterCurves(whichColumn=1):
    """
    Loads SDSS filter curves from a standard location within this module. The default whichColumn=1
    corresponds to curves of the quantum efficiency on the sky looking through 1.3 airmasses at APO
    for a point source. Values of 2-4 are also possible but probably not what you want. Consult the
    filter data file headers for details.
    """
    # Get the path that this module was loaded from.
    import driver
    myPath = os.path.dirname(os.path.abspath(driver.__file__))
    # Build the path where the filter curves should be.
    filterPath = os.path.join(myPath,'data','throughput')
    curves = { }
    for band in 'ugriz':
        filterData = np.loadtxt(os.path.join(filterPath,'sdss_jun2001_%s_atm.dat' % band),unpack=True)
        curves[band] = WavelengthFunction(filterData[0],filterData[whichColumn],extrapolatedValue=0.)
    return curves

if __name__ == '__main__':
    # Run some tests on classes defined here.
    wave = np.arange(4000.,10000.,10.)
    flux = np.exp(-0.5*(wave-7000.)**2/1000.**2)
    spec = SpectralFluxDensity(wave,flux,extrapolatedValue=0.)
    assert spec.wavelength.base is wave,"Numpy wavelength array should not be copied"
    print(spec.getABMagnitudes())
    spec2 = spec.createRescaled('g',22.75)
    print(spec2.getABMagnitudes())
    spec3 = spec.createRedshifted(1.)
    spec4 = spec.createRedshifted(1.,preserveWavelengths=True)
