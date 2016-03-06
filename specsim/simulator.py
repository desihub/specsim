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
import astropy.table

import specsim.config
import specsim.atmosphere
import specsim.instrument
import specsim.source


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

        # Initalize our component models.
        self.atmosphere = specsim.atmosphere.initialize(config)
        self.instrument = specsim.instrument.initialize(config)
        self.source = specsim.source.initialize(config)

        # Initialize our table of simulation results.
        self.camera_names = []
        self.camera_slices = {}
        num_rows = len(config.wavelength)
        flux_unit = u.erg / (u.cm**2 * u.s * u.Angstrom)
        self.simulated = astropy.table.Table(
            meta=dict(description='Specsim simulation results'))
        self.simulated.add_column(astropy.table.Column(
            name='wavelength', data=config.wavelength))
        self.simulated.add_column(astropy.table.Column(
            name='source_flux', dtype=float, length=num_rows, unit=flux_unit))
        self.simulated.add_column(astropy.table.Column(
            name='source_fiber_flux', dtype=float, length=num_rows,
            unit=flux_unit))
        self.simulated.add_column(astropy.table.Column(
            name='sky_fiber_flux', dtype=float, length=num_rows,
            unit=flux_unit))
        self.simulated.add_column(astropy.table.Column(
            name='num_source_photons', dtype=float, length=num_rows))
        self.simulated.add_column(astropy.table.Column(
            name='num_sky_photons', dtype=float, length=num_rows))
        for camera in self.instrument.cameras:
            name = camera.name
            self.camera_names.append(name)
            self.camera_slices[name] = camera.ccd_slice
            self.simulated.add_column(astropy.table.Column(
                name='num_source_electrons_{0}'.format(name),
                dtype=float, length=num_rows))
            self.simulated.add_column(astropy.table.Column(
                name='num_sky_electrons_{0}'.format(name),
                dtype=float, length=num_rows))
            self.simulated.add_column(astropy.table.Column(
                name='num_dark_electrons_{0}'.format(name),
                dtype=float, length=num_rows))
            self.simulated.add_column(astropy.table.Column(
                name='read_noise_electrons_{0}'.format(name),
                dtype=float, length=num_rows))

        # Initialize each camera's table of results downsampled to
        # output pixels.
        self.camera_output = []
        for camera in self.instrument.cameras:
            table = astropy.table.Table(
                meta=dict(description='{0}-camera output'.format(camera.name)))
            num_rows = len(camera.output_wavelength)
            table.add_column(astropy.table.Column(
                name='wavelength', data=camera.output_wavelength))
            table.add_column(astropy.table.Column(
                name='num_source_electrons', dtype=float, length=num_rows))
            table.add_column(astropy.table.Column(
                name='num_sky_electrons', dtype=float, length=num_rows))
            table.add_column(astropy.table.Column(
                name='num_dark_electrons', dtype=float, length=num_rows))
            table.add_column(astropy.table.Column(
                name='read_noise_electrons', dtype=float, length=num_rows))
            table.add_column(astropy.table.Column(
                name='random_noise_electrons', dtype=float, length=num_rows))
            table.add_column(astropy.table.Column(
                name='variance_electrons', dtype=float, length=num_rows))
            table.add_column(astropy.table.Column(
                name='flux_calibration', dtype=float, length=num_rows,
                unit=flux_unit))
            table.add_column(astropy.table.Column(
                name='observed_flux', dtype=float, length=num_rows,
                unit=flux_unit))
            table.add_column(astropy.table.Column(
                name='flux_inverse_variance', dtype=float, length=num_rows,
                unit=flux_unit ** -2))
            self.camera_output.append(table)


    def simulate(self):
        """Simulate a single exposure.
        """
        # Get references to our results columns.
        wavelength = self.simulated['wavelength']
        source_flux = self.simulated['source_flux']
        source_fiber_flux = self.simulated['source_fiber_flux']
        sky_fiber_flux = self.simulated['sky_fiber_flux']
        num_source_photons = self.simulated['num_source_photons']
        num_sky_photons = self.simulated['num_sky_photons']

        # Get the source flux incident on the atmosphere.
        source_flux[:] = self.source.flux_out.to(source_flux.unit)

        # Calculate the source flux entering a fiber.
        source_fiber_flux[:] = (
            source_flux *
            self.atmosphere.extinction *
            self.instrument.get_fiber_acceptance(self.source)
            ).to(source_fiber_flux.unit)

        # Calculate the sky flux entering a fiber.
        sky_fiber_flux[:] = (
            self.atmosphere.surface_brightness *
            self.instrument.fiber_area
            ).to(sky_fiber_flux.unit)

        # Calculate the mean number of source photons entering the fiber
        # per simulation bin.
        num_source_photons[:] = (
            source_fiber_flux *
            self.instrument.photons_per_bin *
            self.instrument.exposure_time
            ).to(1).value

        # Calculate the mean number of sky photons entering the fiber
        # per simulation bin.
        num_sky_photons[:] = (
            sky_fiber_flux *
            self.instrument.photons_per_bin *
            self.instrument.exposure_time
            ).to(1).value

        # Calculate the calibration from constant unit source flux above
        # the atmosphere to number of source photons entering the fiber.
        # We use this below to calculate the flux inverse variance in
        # each camera.
        source_flux_to_photons = (
            self.atmosphere.extinction *
            self.instrument.get_fiber_acceptance(self.source) *
            self.instrument.photons_per_bin *
            self.instrument.exposure_time).to(source_flux.unit ** -1).value

        # Loop over cameras to calculate their individual responses.
        for output, camera in zip(self.camera_output, self.instrument.cameras):

            # Get references to this camera's columns.
            num_source_electrons = self.simulated[
                'num_source_electrons_{0}'.format(camera.name)]
            num_sky_electrons = self.simulated[
                'num_sky_electrons_{0}'.format(camera.name)]
            num_dark_electrons = self.simulated[
                'num_dark_electrons_{0}'.format(camera.name)]
            read_noise_electrons = self.simulated[
                'read_noise_electrons_{0}'.format(camera.name)]

            # Calculate the mean number of source electrons detected in the CCD.
            num_source_electrons[:] = camera.apply_resolution(
                num_source_photons * camera.throughput)

            # Calculate the mean number of sky electrons detected in the CCD.
            num_sky_electrons[:] = camera.apply_resolution(
                num_sky_photons * camera.throughput)

            # Calculate the mean number of dark current electrons in the CCD.
            num_dark_electrons[:] = (
                camera.dark_current_per_bin *
                self.instrument.exposure_time).to(u.electron).value

            # Copy the read noise in units of electrons.
            read_noise_electrons[:] = (
                camera.read_noise_per_bin.to(u.electron).value)

            # Calculate the corresponding downsampled output quantities.
            output['num_source_electrons'] = (
                camera.downsample(num_source_electrons))
            output['num_sky_electrons'] = (
                camera.downsample(num_sky_electrons))
            output['num_dark_electrons'] = (
                camera.downsample(num_dark_electrons))
            output['read_noise_electrons'] = np.sqrt(
                camera.downsample(read_noise_electrons ** 2))
            output['variance_electrons'] = (
                output['num_source_electrons'] +
                output['num_sky_electrons'] +
                output['num_dark_electrons'] +
                output['read_noise_electrons'] ** 2)

            # Calculate the effective calibration from detected electrons to
            # source flux above the atmosphere, downsampled to output pixels.
            output['flux_calibration'] = 1.0 / camera.downsample(
                camera.apply_resolution(
                    camera.throughput * source_flux_to_photons))

            # Calculate the calibrated flux in this camera.
            output['observed_flux'] = (
                output['flux_calibration'] * output['num_source_electrons'])

            # Calculate the corresponding flux inverse variance.
            output['flux_inverse_variance'] = (
                output['flux_calibration'] ** -2 *
                output['variance_electrons'] ** -1)

            # Zero our random noise realization column.
            output['random_noise_electrons'][:] = 0.


    def generate_random_noise(self, random_state=None):
        """Generate a random noise realization for the most recent simulation.

        Fills the "random_noise_electrons" column in each camera's output
        table, which is zeroed after each call to :meth:`simulate`. Can be
        called repeatedly for the same simulated response to generate different
        noise realizations.

        Noise is modeled as a Poisson fluctuation of the mean number of detected
        electrons from the source + sky + dark current, combined with a
        Gaussian fluctuation of the mean read noise.

        The noise is generated in units of detected electrons.  To propagate
        the generated noise to a corresponding calibrated flux noise, use::

            output['flux_calibration'] * output['random_noise_electrons']

        Parameters
        ----------
        random_state : numpy.random.RandomState or None
            The random number generation state to use for reproducible noise
            realizations. A new state will be created with a randomized seed
            if None is specified.
        """
        if random_state is None:
            random_state = np.random.RandomState()

        for output in self.camera_output:
            mean_electrons = (
                output['num_source_electrons'] +
                output['num_sky_electrons'] + output['num_dark_electrons'])
            output['random_noise_electrons'] = (
                random_state.poisson(mean_electrons) - mean_electrons +
                random_state.normal(
                    scale=output['read_noise_electrons'], size=len(output)))


    def plot(self):
        """Plot results of the last simulation.

        Uses the contents of the :attr:`simulated` and :attr:`camera_output`
        astropy tables to plot the results of the last call to :meth:`simulate`.
        """
        pass
