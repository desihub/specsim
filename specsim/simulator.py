# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Top-level manager for spectroscopic simulation.

A simulator is usually initialized from a configuration, for example:

    >>> simulator = Simulator('test')

See :doc:`/api` for examples of changing model parameters defined in the
configuration.  Certain parameters can also be changed after a model has
been initialized, for example:

    >>> simulator.atmosphere.airmass = 1.5
    >>> simulator.instrument.exposure_time = 1200 * u.s

See :mod:`source`, :mod:`atmosphere` and :mod:`instrument` for details.
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
    """Manage the simulation of a source, atmosphere and instrument.

    A simulator has no configuration parameters of its own.

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
        self._simulated = astropy.table.Table(
            meta=dict(description='Specsim simulation results'))
        self._simulated.add_column(astropy.table.Column(
            name='wavelength', data=config.wavelength))
        self._simulated.add_column(astropy.table.Column(
            name='source_flux', dtype=float, length=num_rows, unit=flux_unit))
        self._simulated.add_column(astropy.table.Column(
            name='source_fiber_flux', dtype=float, length=num_rows,
            unit=flux_unit))
        self._simulated.add_column(astropy.table.Column(
            name='sky_fiber_flux', dtype=float, length=num_rows,
            unit=flux_unit))
        self._simulated.add_column(astropy.table.Column(
            name='num_source_photons', dtype=float, length=num_rows))
        self._simulated.add_column(astropy.table.Column(
            name='num_sky_photons', dtype=float, length=num_rows))
        for camera in self.instrument.cameras:
            name = camera.name
            self.camera_names.append(name)
            self.camera_slices[name] = camera.ccd_slice
            self._simulated.add_column(astropy.table.Column(
                name='num_source_electrons_{0}'.format(name),
                dtype=float, length=num_rows))
            self._simulated.add_column(astropy.table.Column(
                name='num_sky_electrons_{0}'.format(name),
                dtype=float, length=num_rows))
            self._simulated.add_column(astropy.table.Column(
                name='num_dark_electrons_{0}'.format(name),
                dtype=float, length=num_rows))
            self._simulated.add_column(astropy.table.Column(
                name='read_noise_electrons_{0}'.format(name),
                dtype=float, length=num_rows))

        # Initialize each camera's table of results downsampled to
        # output pixels.
        self._camera_output = []
        for camera in self.instrument.cameras:
            meta = dict(
                name=camera.name,
                pixel_size=camera.output_pixel_size)
            table = astropy.table.Table(meta=meta)
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
            self._camera_output.append(table)


    @property
    def simulated(self):
        """astropy.table.Table: Table of high-resolution simulation results.

        This table is tabulated using the high-resolution wavelength used for
        internal calclulations and overwritten during each call to
        :meth:`simulate`.  See :doc:`/output` for details of this table's
        contents.
        """
        return self._simulated


    @property
    def camera_output(self):
        """list: List of per-camera simulation output tables.

        Tables are listed in order of increasing wavelength and tabulated
        using the output pixels defined for each camera.  Tables are overwritten
        during each call to :meth:`simulate`.  See :doc:`/output` for details
        of the contents of each table in this list.
        """
        return self._camera_output


    def simulate(self):
        """Simulate a single exposure.

        Simulation results are written to internal tables that are overwritten
        each time this method is called.
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


    def plot(self, title=None):
        """Plot results of the last simulation.

        Uses the contents of the :attr:`simulated` and :attr:`camera_output`
        astropy tables to plot the results of the last call to :meth:`simulate`.
        See :func:`plot_simulation` for details.

        Parameters
        ----------
        title : str or None
            Plot title to use.  If None is specified, a title will be
            automatically generated using the source name, airmass and
            exposure time.
        """
        if title is None:
            title = (
                '{0}, X={1}, t={2}'
                .format(self.source.name, self.atmosphere.airmass,
                        self.instrument.exposure_time))
        plot_simulation(self.simulated, self.camera_output, title)


def plot_simulation(simulated, camera_output, title=None,
                    min_electrons=2.5, figsize=(11, 8.5), label_size='medium'):
    """Plot simulation output tables.

    This function is normally called via :meth:`Simulator.plot` but is provided
    separately so that plots can be generated from results saved to a file.

    Use :meth:`show <matplotlib.pyplot.show` and :meth:`savefig
    <matplotlib.pyplot.savefig>` to show or save the resulting plot.

    See :doc:`/cmdline` for a sample plot.

    Requires that the matplotlib package is installed.

    Parameters
    ----------
    simulated : astropy.table.Table
        Simulation results on the high-resolution simulation wavelength grid.
    camera_output : list
        Lists of tables of per-camera simulation results tabulated on each
        camera's output pixel grid.
    title : str or None
        Descriptive title to use for the plot.
    min_electrons : float
        Minimum y-axis value for displaying numbers of detected electrons.
    figsize : tuple
        Tuple (width, height) specifying the figure size to use in inches.
        See :meth:`matplotlib.pyplot.subplots` for details.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    if title is not None:
        ax1.set_title(title)

    wave = simulated['wavelength']
    dwave = np.gradient(wave)
    waveunit = '{0:Generic}'.format(wave.unit)
    fluxunit = '{0:Generic}'.format(simulated['source_flux'].unit)

    # Plot fluxes above the atmosphere and into the fiber.

    src_flux = simulated['source_flux']
    src_fiber_flux = simulated['source_fiber_flux']
    sky_fiber_flux = simulated['sky_fiber_flux']

    ymin, ymax = 0.1 * np.min(src_flux), 10. * np.max(src_flux)

    line, = ax1.plot(wave, src_flux, 'r-')
    ax1.fill_between(wave, src_fiber_flux + sky_fiber_flux,
                     ymin, color='b', alpha=0.2, lw=0)
    ax1.fill_between(wave, src_fiber_flux, ymin, color='r', alpha=0.2, lw=0)

    # This kludge is because the label arg to fill_between() does not
    # propagate to legend() in matplotlib < 1.5.
    sky_fill = Rectangle((0, 0), 1, 1, fc='b', alpha=0.2)
    src_fill = Rectangle((0, 0), 1, 1, fc='r', alpha=0.2)
    ax1.legend(
        (line, sky_fill, src_fill),
        ('Source above atmosphere', 'Sky into fiber', 'Source into fiber'),
        loc='best', fancybox=True, framealpha=0.5, ncol=3, fontsize=label_size)

    ax1.set_ylim(ymin, ymax)
    ax1.set_yscale('log')
    ax1.set_ylabel('Flux [{0}]'.format(fluxunit))

    # Plot numbers of photons into the fiber.

    nsky = simulated['num_sky_photons'] / dwave
    nsrc = simulated['num_source_photons'] / dwave
    nmax = np.max(nsrc)

    ax2.fill_between(wave, nsky + nsrc, 1e-1 * nmax, color='b', alpha=0.2, lw=0)
    ax2.fill_between(wave, nsrc, 1e-1 * nmax, color='r', alpha=0.2, lw=0)

    ax2.legend(
        (sky_fill, src_fill),
        ('Sky into fiber', 'Source into fiber'),
        loc='best', fancybox=True, framealpha=0.5, ncol=2, fontsize=label_size)

    ax2.set_ylim(1e-1 * nmax, 10. * nmax)
    ax2.set_yscale('log')
    ax2.set_ylabel('Mean photons / {0}'.format(waveunit))
    ax2.set_xlim(wave[0], wave[-1])

    # Plot numbers of electrons detected by each CCD.

    for output in camera_output:

        cwave = output['wavelength']
        dwave = np.gradient(cwave)
        nsky = output['num_sky_electrons'] / dwave
        nsrc = output['num_source_electrons'] / dwave
        ndark = output['num_dark_electrons'] / dwave
        read_noise = output['read_noise_electrons'] / np.sqrt(dwave)
        total_noise = np.sqrt(output['variance_electrons'] / dwave)
        nmax = max(nmax, np.max(nsrc))

        ax3.fill_between(
            cwave, ndark + nsky + nsrc, min_electrons, color='b',
            alpha=0.2, lw=0)
        ax3.fill_between(
            cwave, ndark + nsrc, min_electrons, color='r', alpha=0.2, lw=0)
        ax3.fill_between(
            cwave, ndark, min_electrons, color='k', alpha=0.2, lw=0)
        ax3.scatter(cwave, total_noise, color='k', lw=0., s=0.5, alpha=0.5)
        line2, = ax3.plot(cwave, read_noise, color='k', ls='--', alpha=0.5)

    # This kludge is because the label arg to fill_between() does not
    # propagate to legend() in matplotlib < 1.5.
    line1, = ax3.plot([], [], 'k-')
    dark_fill = Rectangle((0, 0), 1, 1, fc='k', alpha=0.2)
    ax3.legend(
        (sky_fill, src_fill, dark_fill, line1, line2),
        ('Sky detected', 'Source detected', 'Dark current',
         'RMS total noise', 'RMS read noise'),
        loc='best', fancybox=True, framealpha=0.5, ncol=5, fontsize=label_size)

    ax3.set_ylim(min_electrons, 2e2 * min_electrons)
    ax3.set_yscale('log')
    ax3.set_ylabel('Mean electrons / {0}'.format(waveunit))
    ax3.set_xlim(wave[0], wave[-1])
    ax3.set_xlabel('Wavelength [{0}]'.format(waveunit))

    # Remove x-axis ticks on the upper panels.
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
