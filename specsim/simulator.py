# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Top-level manager for spectroscopic simulation.

For an overview of using a :class:`Simulator`, see the
`examples notebook
<https://github.com/desihub/specsim/blob/master/docs/nb/SimulationExamples.ipynb>`__.

A simulator is usually initialized from a configuration, for example:

    >>> simulator = Simulator('test', num_fibers=500)

See :doc:`/api` for examples of changing model parameters defined in the
configuration.  Certain parameters can also be changed after a model has
been initialized, for example:

    >>> simulator.atmosphere.airmass = 1.5
    >>> simulator.observation.exposure_time = 1200 * u.s

See :mod:`source`, :mod:`atmosphere` and :mod:`instrument` for details.

The positions and properties of individual sources in an exposure can be
specified using optional array arguments to the :meth:`simulate method
<Simulator.simulate>`.
"""
from __future__ import print_function, division

import math
import os.path

import numpy as np
import scipy.sparse as sp

from astropy import units as u
import astropy.table
import astropy.io.fits

import specsim.config
import specsim.atmosphere
import specsim.instrument
import specsim.source
import specsim.fiberloss
import specsim.observation


class Simulator(object):
    """Manage the simulation of a source, atmosphere and instrument.

    A simulator has no configuration parameters of its own.

    Parameters
    ----------
    config : specsim.config.Configuration or str
        A configuration object or configuration name.
    num_fibers : int
        Number of fibers to simulate.
    camera_output : bool
        Include per-camera output tables in simulation results when True.
        When this is False, our ``camera_output`` attribute will return an
        empty list and the ``num_source_electrons_*`` columns in our
        ``simulated`` table will not be resolution convolved.
        Setting this parameter to False will save memory and time when
        per-camera outputs are not needed.
    verbose : bool
        Print information about the simulation progress.
    """
    def __init__(self, config, num_fibers=2, camera_output=True, verbose=False):

        if specsim.config.is_string(config):
            config = specsim.config.load_config(config)
        config.verbose = verbose
        self.verbose = verbose

        # Initalize our component models.
        self.atmosphere = specsim.atmosphere.initialize(config)
        self.instrument = specsim.instrument.initialize(config, camera_output)
        self.source = specsim.source.initialize(config)
        self.observation = specsim.observation.initialize(config)

        self._num_fibers = int(num_fibers)
        if self._num_fibers < 1:
            raise ValueError('Must have num_fibers >= 1.')

        # Initialize our table of simulation results.
        self.camera_names = []
        self.camera_slices = {}
        num_rows = len(config.wavelength)
        shape = (self.num_fibers,)
        column_args = dict(dtype=float, length=num_rows, shape=shape)
        flux_unit = u.erg / (u.cm**2 * u.s * u.Angstrom)
        self._simulated = astropy.table.Table(
            meta=dict(description='Specsim simulation results'))
        self._simulated.add_column(astropy.table.Column(
            name='wavelength', data=config.wavelength))
        self._simulated.add_column(astropy.table.Column(
            name='source_flux', unit=flux_unit, **column_args))
        self._simulated.add_column(astropy.table.Column(
            name='fiberloss', **column_args))
        self._simulated.add_column(astropy.table.Column(
            name='source_fiber_flux', unit=flux_unit, **column_args))
        self._simulated.add_column(astropy.table.Column(
            name='sky_fiber_flux', unit=flux_unit, **column_args))
        self._simulated.add_column(astropy.table.Column(
            name='num_source_photons', **column_args))
        self._simulated.add_column(astropy.table.Column(
            name='num_sky_photons', **column_args))
        for camera in self.instrument.cameras:
            name = camera.name
            self.camera_names.append(name)
            self.camera_slices[name] = camera.ccd_slice
            self._simulated.add_column(astropy.table.Column(
                name='num_source_electrons_{0}'.format(name), **column_args))
            self._simulated.add_column(astropy.table.Column(
                name='num_sky_electrons_{0}'.format(name), **column_args))
            self._simulated.add_column(astropy.table.Column(
                name='num_dark_electrons_{0}'.format(name), **column_args))
            self._simulated.add_column(astropy.table.Column(
                name='read_noise_electrons_{0}'.format(name), **column_args))

        # Count the number of bytes used in the simulated table.
        self.table_bytes = 0
        for name in self._simulated.colnames:
            d = self._simulated[name].data
            self.table_bytes += np.prod(d.shape) * d.dtype.itemsize

        # Initialize each camera's table of results downsampled to
        # output pixels, if requested.
        self._camera_output = []
        if camera_output:
            for camera in self.instrument.cameras:
                meta = dict(
                    name=camera.name, num_fibers=self.num_fibers,
                    pixel_size=camera.output_pixel_size)
                table = astropy.table.Table(meta=meta)
                column_args['length'] = len(camera.output_wavelength)
                table.add_column(astropy.table.Column(
                    name='wavelength', data=camera.output_wavelength))
                table.add_column(astropy.table.Column(
                    name='num_source_electrons', **column_args))
                table.add_column(astropy.table.Column(
                    name='num_sky_electrons', **column_args))
                table.add_column(astropy.table.Column(
                    name='num_dark_electrons', **column_args))
                table.add_column(astropy.table.Column(
                    name='read_noise_electrons', **column_args))
                table.add_column(astropy.table.Column(
                    name='random_noise_electrons', **column_args))
                table.add_column(astropy.table.Column(
                    name='variance_electrons', **column_args))
                table.add_column(astropy.table.Column(
                    name='flux_calibration', **column_args))
                table.add_column(astropy.table.Column(
                    name='observed_flux', unit=flux_unit, **column_args))
                table.add_column(astropy.table.Column(
                    name='flux_inverse_variance', unit=flux_unit ** -2,
                    **column_args))
                # Add bytes used in this table to our running total.
                for name in table.colnames:
                    d = table[name].data
                    self.table_bytes += np.prod(d.shape) * d.dtype.itemsize

                self._camera_output.append(table)

        if self.verbose:
            print('Allocated {0:.1f}Mb of table data.'
                  .format(self.table_bytes / (2. ** 20)))

    @property
    def num_fibers(self):
        """Number of fibers being simulated.
        """
        return self._num_fibers

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

        Returns an empty list if this Simulator was initialized with
        ``camera_output`` False.
        """
        return self._camera_output

    def simulate(self, sky_positions=None, focal_positions=None,
                 fiber_acceptance_fraction=None,
                 source_fluxes=None, source_types=None, source_fraction=None,
                 source_half_light_radius=None,
                 source_minor_major_axis_ratio=None, source_position_angle=None,
                 calibration_surface_brightness=None, save_fiberloss=None):
        """Simulate a single exposure.

        Simulation results are written to internal tables that are overwritten
        each time this method is called.  Some metadata is also saved as
        attributes of this object: `focal_x`, `focal_y`, `fiber_area`.

        The positions and properties of each source can optionally be specified
        individually for each fiber via array arguments.  Any parameters that
        are not specified this way will use the same value for each fiber
        taken from the configuration data, as noted below.

        Fibers are positioned using either (x,y) focal-plane coordinates
        or else (ra,dec) sky coordinates.  The first available item on this
        list will be used:
        - ``focal_positions`` argument.
        - ``sky_positions`` argument.
        - ``source.location.constants.focal_x,y`` config data.
        - ``source.location.sky`` config data.
        When config data is used, it is duplicated for all fibers. Use the
        verbose mode to see details on how fibers are being positioned. Note
        that the observing airmass will be calculated when positioning with
        sky coordinates, and then available via ``self.observation.airmass``.

        Calibration exposures can be simulated by providing an array of
        ``calibration_surface_brightness`` values to use.  In this case,
        the source and fiberloss inputs are ignored, and no atmospheric
        emission or extinction are applied.

        Per-camera output tables will not be filled if this Simulator was
        initialized with ``camera_output`` False.

        Parameters
        ----------
        sky_positions : astropy.units.Quantity or None
            Sky positions of each object. Must have a length equal to
            num_fibers.  Defaults to ``source.location.sky`` when None.
        focal_positions : astropy.units.Quantity or None
            Focal-plane coordinates of each object relative to the plate
            center, with length units. Must have a length equal to num_fibers.
            Defaults to ``source.location.constants.focal_x,y`` when None.
        fiber_acceptance_fraction : array or None
            Array of shape (num_fibers, num_wlen) giving the fiber acceptance
            fraction to use for each fiber. Defaults to calling
            :meth:`fiberloss.calculate_fiber_acceptance_fraction` when None.
        source_fluxes : array or None
            Array of shape (num_fibers, num_wlen) giving the source flux
            above the atmosphere illuminating each fiber. Defaults to
            ``source.table`` when None.
        source_types : array or None
            Array of strings with length num_fibers.  Each string must have a
            corresponding pre-loaded fiberloss file in the configuration.
            Defaults to ``source.type`` when None.
        source_fraction : array or None
            Array of shape (num_fibers, 2) giving the disk and bulge fractions
            for each source.  Fractions must be in the range [0, 1] and their
            sum must be <= 1.  If their sum is <1, the remainder is modeled as
            a point-like component.  Defaults to
            ``source.profile.disk,bulge_fraction`` when None.
        source_half_light_radius : array or None
            Array of shape (num_fibers, 2) giving the disk and bulge half-light
            radii in on-sky angular units.  Defaults to values in
            ``source.profile.disk,bulge_shape`` when None.
        source_minor_major_axis_ratio : array or None
            Array of shape (num_fibers, 2) giving the disk and bulge minor/major
            axis ratios, in the range (0,1]. Defaults to values in
            ``source.profile.disk,bulge_shape`` when None.
        source_position_angle : array or None
            Array of shape (num_fibers, 2) giving the disk and bulge major
            axis alignments, expressed as a clockwise rotation from the +x
            axis, with angular units. Defaults to values in
            ``source.profile.disk,bulge_shape`` when None.
        calibration_surface_brightness : array or None
            Array of shape (num_fibers, num_wlen) giving the calibration
            source surface brightness illuminating each fiber. When this is
            set, all source parameters (those beginning with ``source_``)
            and ``fiber_acceptance_fraction`` are ignored and this is assumed
            to be a calibration exposure.
        save_fiberloss : str or None
            Basename for saving FITS images and tabulated fiberloss.
            Ignored unless instrument.fiberloss.method is galsim.
        """
        # Get references to our results columns. Since table rows index
        # wavelength, the shape of each column is (nwlen, nfiber) and
        # therefore some transposes are necessary to match with the shape
        # (nfiber, nwlen) of source_fluxes and fiber_acceptance_fraction,
        # and before calling the camera downsample() and apply_resolution()
        # methods (which expect wavelength in the last index).
        wavelength = self.simulated['wavelength']
        source_flux = self.simulated['source_flux']
        fiberloss = self.simulated['fiberloss']
        source_fiber_flux = self.simulated['source_fiber_flux']
        sky_fiber_flux = self.simulated['sky_fiber_flux']
        num_source_photons = self.simulated['num_source_photons']
        num_sky_photons = self.simulated['num_sky_photons']
        nwlen = len(wavelength)

        # Is this a calibration exposure?
        calibrating = calibration_surface_brightness is not None

        # Position each fiber.
        if focal_positions is not None:
            if len(focal_positions) != self.num_fibers:
                raise ValueError(
                    'Expected {0:d} focal_positions.'.format(self.num_fibers))
            try:
                focal_positions = focal_positions.to(u.mm)
            except (AttributeError, u.UnitConversionError):
                raise ValueError('Invalid units for focal_positions.')
            self.focal_x, self.focal_y = focal_positions.T
            on_sky = False
            if self.verbose:
                print('Fibers positioned with focal_positions array.')
        elif sky_positions is not None:
            if len(sky_positions) != self.num_fibers:
                raise ValueError(
                    'Expected {0:d} sky_positions.'.format(self.num_fibers))
            self.focal_x, self.focal_y = self.observation.locate_on_focal_plane(
                sky_positions, self.instrument)
            on_sky = True
            if self.verbose:
                print('Fibers positioned with sky_positions array.')
        elif self.source.focal_xy is not None:
            self.focal_x, self.focal_y = np.tile(
                self.source.focal_xy, [self.num_fibers, 1]).T
            on_sky = False
            if self.verbose:
                print('All fibers positioned at config (x,y).')
        elif self.source.sky_position is not None:
            focal_x, focal_y = self.observation.locate_on_focal_plane(
                self.source.sky_position, self.instrument)
            self.focal_x = np.tile(focal_x, [self.num_fibers])
            self.focal_y = np.tile(focal_y, [self.num_fibers])
            on_sky = True
            if self.verbose:
                print('All fibers positioned at config (ra,dec).')
        else:
            raise RuntimeError('No fiber positioning info available.')

        if not calibrating and on_sky:
            # Set the observing airmass in the atmosphere model using
            # Eqn.3 of Krisciunas & Schaefer 1991.
            obs_zenith = 90 * u.deg - self.observation.boresight_altaz.alt
            obs_airmass = (1 - 0.96 * np.sin(obs_zenith) ** 2) ** -0.5
            self.atmosphere.airmass = obs_airmass
            if self.verbose:
                print('Calculated alt={0:.1f} az={1:.1f} airmass={2:.3f}'
                      .format(self.observation.boresight_altaz.alt,
                              self.observation.boresight_altaz.az, obs_airmass))

        # Check that all sources are within the field of view.
        focal_r = np.sqrt(self.focal_x ** 2 + self.focal_y ** 2)
        if np.any(focal_r > self.instrument.field_radius):
            raise RuntimeError(
                'A source is located outside the field of view: r > {0:.1f}'
                .format(self.instrument.field_radius))

        # Calculate the on-sky fiber areas at each focal-plane location.
        radial_fiber_size = (
            0.5 * self.instrument.fiber_diameter /
            self.instrument.radial_scale(focal_r))
        azimuthal_fiber_size = (
            0.5 * self.instrument.fiber_diameter /
            self.instrument.azimuthal_scale(focal_r))
        self.fiber_area = np.pi * radial_fiber_size * azimuthal_fiber_size

        if calibrating:
            # Convert surface brightness to flux entering each fiber.
            if calibration_surface_brightness.shape != (self.num_fibers, nwlen):
                raise ValueError(
                    'Invalid shape for calibration_surface_brightness.')
            try:
                source_flux[:] = (
                    calibration_surface_brightness.T * self.fiber_area).to(
                        source_flux.unit)
            except AttributeError:
                raise ValueError(
                    'Missing units for calibration_surface_brightness.')
            except u.UnitConversionError:
                raise ValueError(
                    'Invalid units for calibration_surface_brightness.')
            # Fiberloss is one.
            fiberloss[:] = 1.
            # No atmospheric extinction of calibration sources.
            source_fiber_flux[:] = source_flux.to(source_fiber_flux.unit)
            # No sky emission added to calibration sources.
            sky_fiber_flux[:] = 0.
            # Calibration from constant source flux entering the fiber to
            # number of source photons entering the fiber.
            source_flux_to_photons = (
                self.instrument.photons_per_bin[:, np.newaxis] *
                self.observation.exposure_time
                ).to(source_flux.unit ** -1).value
        else:
            # Get the source fluxes incident on the atmosphere.
            if source_fluxes is None:
                source_fluxes = self.source.flux_out.to(
                    source_flux.unit)[np.newaxis, :]
            elif source_fluxes.shape != (self.num_fibers, nwlen):
                raise ValueError('Invalid shape for source_fluxes.')
            try:
                source_flux[:] = source_fluxes.to(source_flux.unit).T
            except AttributeError:
                raise ValueError('Missing units for source_fluxes.')
            except u.UnitConversionError:
                raise ValueError('Invalid units for source_fluxes.')

            # Calculate fraction of source illumination entering the fiber.
            if save_fiberloss is not None:
                saved_images_file = save_fiberloss + '.fits'
                saved_table_file = save_fiberloss + '.ecsv'
            else:
                saved_images_file, saved_table_file = None, None

            if fiber_acceptance_fraction is None:
                # Calculate fiberloss using the method specified in
                # instrument.fiberloss_method.
                fiber_acceptance_fraction =\
                    specsim.fiberloss.calculate_fiber_acceptance_fraction(
                        self.focal_x, self.focal_y, wavelength.quantity,
                        self.source, self.atmosphere, self.instrument,
                        source_types, source_fraction, source_half_light_radius,
                        source_minor_major_axis_ratio, source_position_angle,
                        saved_images_file=saved_images_file,
                        saved_table_file=saved_table_file)
            else:
                fiber_acceptance_fraction = np.asarray(
                    fiber_acceptance_fraction)
                if fiber_acceptance_fraction.shape != (self.num_fibers, nwlen):
                    raise ValueError(
                        'Invalid shape for fiber_acceptance_fraction.')
            fiberloss[:] = fiber_acceptance_fraction.T

            # Calculate the source flux entering a fiber.
            source_fiber_flux[:] = (
                source_flux *
                self.atmosphere.extinction[:, np.newaxis] *
                fiberloss
                ).to(source_fiber_flux.unit)

            # Calculate the sky flux entering a fiber.
            sky_fiber_flux[:] = (
                self.atmosphere.surface_brightness[:, np.newaxis] *
                self.fiber_area
                ).to(sky_fiber_flux.unit)

            # Calculate the calibration from constant unit source flux above
            # the atmosphere to number of source photons entering the fiber.
            # We use this below to calculate the flux inverse variance in
            # each camera.
            source_flux_to_photons = (
                self.atmosphere.extinction[:, np.newaxis] *
                fiberloss *
                self.instrument.photons_per_bin[:, np.newaxis] *
                self.observation.exposure_time
                ).to(source_flux.unit ** -1).value

        # Calculate the mean number of source photons entering the fiber
        # per simulation bin.
        num_source_photons[:] = (
            source_fiber_flux *
            self.instrument.photons_per_bin[:, np.newaxis] *
            self.observation.exposure_time
            ).to(1).value

        # Calculate the mean number of sky photons entering the fiber
        # per simulation bin.
        num_sky_photons[:] = (
            sky_fiber_flux *
            self.instrument.photons_per_bin[:, np.newaxis] *
            self.observation.exposure_time
            ).to(1).value

        # Calculate the high-resolution inputs to each camera.
        for camera in self.instrument.cameras:
            # Get references to this camera's columns.
            num_source_electrons = self.simulated[
                'num_source_electrons_{0}'.format(camera.name)]
            num_sky_electrons = self.simulated[
                'num_sky_electrons_{0}'.format(camera.name)]
            num_dark_electrons = self.simulated[
                'num_dark_electrons_{0}'.format(camera.name)]
            read_noise_electrons = self.simulated[
                'read_noise_electrons_{0}'.format(camera.name)]

            # Calculate the mean number of source electrons detected in the CCD
            # without any resolution applied.
            num_source_electrons[:] = (
                num_source_photons * camera.throughput[:, np.newaxis])

            # Calculate the mean number of sky electrons detected in the CCD
            # without any resolution applied.
            num_sky_electrons[:] = (
                num_sky_photons * camera.throughput[:, np.newaxis])

            # Calculate the mean number of dark current electrons in the CCD.
            num_dark_electrons[:] = (
                camera.dark_current_per_bin[:, np.newaxis] *
                self.observation.exposure_time).to(u.electron).value

            # Copy the read noise in units of electrons.
            read_noise_electrons[:] = (
                camera.read_noise_per_bin[:, np.newaxis].to(u.electron).value)

        if not self.camera_output:
            # All done since no camera output was requested.
            return

        # Loop over cameras to calculate their individual responses
        # with resolution applied and downsampling to output pixels.
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

            # Apply resolution to the source and sky detected electrons on
            # the high-resolution grid.
            num_source_electrons[:] = camera.apply_resolution(
                num_source_electrons)
            num_sky_electrons[:] = camera.apply_resolution(
                num_sky_electrons)

            # Calculate the corresponding downsampled output quantities.
            output['num_source_electrons'][:] = (
                camera.downsample(num_source_electrons))
            output['num_sky_electrons'][:] = (
                camera.downsample(num_sky_electrons))
            output['num_dark_electrons'][:] = (
                camera.downsample(num_dark_electrons))
            output['read_noise_electrons'][:] = np.sqrt(
                camera.downsample(read_noise_electrons ** 2))
            output['variance_electrons'][:] = (
                output['num_source_electrons'] +
                output['num_sky_electrons'] +
                output['num_dark_electrons'] +
                output['read_noise_electrons'] ** 2)

            # Calculate the effective calibration from detected electrons to
            # source flux above the atmosphere, downsampled to output pixels.
            output['flux_calibration'][:] = 1.0 / camera.downsample(
                camera.apply_resolution(
                    source_flux_to_photons * camera.throughput.reshape(-1, 1)))

            # Calculate the calibrated flux in this camera.
            output['observed_flux'][:] = (
                output['flux_calibration'] * output['num_source_electrons'])

            # Calculate the corresponding flux inverse variance.
            output['flux_inverse_variance'][:] = (
                output['flux_calibration'] ** -2 *
                output['variance_electrons'] ** -1)

            # Zero our random noise realization column.
            output['random_noise_electrons'][:] = 0.

    def generate_random_noise(self, random_state=None, use_poisson=True):
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
        use_poisson : bool
            If False, use numpy.random.normal instead of numpy.random.poisson.
            This is useful for simulations where one wants the same noise 
            realization for varying average flux (numpy.random.poisson
            uses a varying number of random numbers depending on the mean).
        """
        if not self.camera_output:
            raise RuntimeError('Simulator initialized with no camera output.')

        if random_state is None:
            random_state = np.random.RandomState()

        for output in self.camera_output:
            mean_electrons = (
                output['num_source_electrons'] +
                output['num_sky_electrons'] + output['num_dark_electrons'])
            if use_poisson :
                output['random_noise_electrons'] = (
                    random_state.poisson(mean_electrons) - mean_electrons +
                    random_state.normal(scale=output['read_noise_electrons']))
            else :
                output['random_noise_electrons'] = random_state.normal(scale=np.sqrt( mean_electrons + output['read_noise_electrons']**2))

    def save(self, filename, clobber=True):
        """Save results of the last simulation to a FITS file.

        Parameters
        ----------
        filename : str
            Name of the file where results should be saved.  Must use the
            .fits extension.
        clobber : bool
            Any existing file will be silently overwritten when clobber is True.
        """
        base, ext = os.path.splitext(filename)
        if ext != '.fits':
            raise ValueError('Filename must have the .fits extension.')
        # Create an empty primary HDU for header keywords
        primary = astropy.io.fits.PrimaryHDU()
        hdr = primary.header
        hdr['name'] = self.instrument.name
        # Save each table to its own HDU.
        simulated = astropy.io.fits.BinTableHDU(
            name='simulated', data=self.simulated.as_array())
        hdus = astropy.io.fits.HDUList([primary, simulated])
        for output in self.camera_output:
            hdus.append(astropy.io.fits.BinTableHDU(
                name=output.meta['name'], data=output.as_array()))
        # Write the file.
        hdus.writeto(filename, clobber=clobber)
        hdus.close()

    def plot(self, fiber=0, wavelength_min=None, wavelength_max=None,
             title=None, min_electrons=2.5):
        """Plot results of the last simulation for a single fiber.

        Uses the contents of the :attr:`simulated` and :attr:`camera_output`
        astropy tables to plot the results of the last call to :meth:`simulate`.
        See :func:`plot_simulation` for details.

        Parameters
        ----------
        fiber : int
            Fiber index to plot.  Must be less than `self.num_fibers`.
        wavelength_min : quantity or None
            Clip the plot below this wavelength, or show the full extent.
        wavelength_max : quantity or None
            Clip the plot above this wavelength, or show the full extent.
        title : str or None
            Plot title to use.  If None is specified, a title will be
            automatically generated using the source name, airmass and
            exposure time.
        """
        if fiber < 0 or fiber >= self.num_fibers:
            raise ValueError('Requested fiber is out of range.')
        if title is None:
            title = (
                'Fiber={0}, X={1}, t={2}'
                .format(fiber, self.atmosphere.airmass,
                        self.observation.exposure_time))
        plot_simulation(self.simulated, self.camera_output, fiber,
                        wavelength_min, wavelength_max, title, min_electrons)


def plot_simulation(simulated, camera_output, fiber=0, wavelength_min=None,
                    wavelength_max=None, title=None, min_electrons=2.5,
                    figsize=(11, 8.5), label_size='medium'):
    """Plot simulation output tables for a single fiber.

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
    fiber : int
        Fiber index to plot.
    wavelength_min : quantity or None
        Clip the plot below this wavelength, or show the full extent.
    wavelength_max : quantity or None
        Clip the plot above this wavelength, or show the full extent.
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

    waveunit = '{0:Generic}'.format(simulated['wavelength'].unit)
    fluxunit = '{0:Generic}'.format(simulated['source_flux'].unit)
    wave = simulated['wavelength'].data
    dwave = np.gradient(wave)

    # Validate the optional wavelength limits and convert to waveunit.
    def validate(name, limit):
        if limit is None:
            return limit
        try:
            return limit.to(waveunit).value
        except AttributeError:
            raise ValueError('Missing unit for {0}.'.format(name))
        except u.UnitConversionError:
            raise ValueError('Invalid unit for {0}.'.format(name))
    wavelength_min = validate('wavelength_min', wavelength_min)
    wavelength_max = validate('wavelength_max', wavelength_max)
    if wavelength_min and wavelength_max and wavelength_min >= wavelength_max:
        raise ValueError('Expected wavelength_min < wavelength_max.')

    # Create a helper function that returns a slice that limits the
    # wavelength array w (with units) to wavelenth_min <= w <= wavelength_max.
    # Returns None if all w < wavelength_min or all w > wavelength_max.
    def get_slice(w):
        assert np.all(np.diff(w) > 0)
        if wavelength_min is None:
            start = 0
        elif wavelength_min > w[-1]:
            return None
        else:
            start = np.where(w >= wavelength_min)[0][0]
        if wavelength_max is None:
            stop = len(w)
        elif wavelength_max < w[0]:
            return None
        else:
            stop = np.where(w <= wavelength_max)[0][-1] + 1
        return slice(start, stop)

    # Trim the full wavelength grid.
    waves = get_slice(wave)
    if waves is None:
        raise ValueError('Wavelength limits do not overlap simulation grid.')
    wave = wave[waves]
    dwave = dwave[waves]

    # Plot fluxes above the atmosphere and into the fiber.

    src_flux = simulated['source_flux'][waves, fiber]
    src_fiber_flux = simulated['source_fiber_flux'][waves, fiber]
    sky_fiber_flux = simulated['sky_fiber_flux'][waves, fiber]

    ymin, ymax = 0.1 * np.min(src_flux), 10. * np.max(src_flux)
    if ymin <= 0:
        # Need ymin > 0 for log scale.
        ymin = 1e-3 * ymax

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

    nsky = simulated['num_sky_photons'][waves, fiber] / dwave
    nsrc = simulated['num_source_photons'][waves, fiber] / dwave
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

        cwave = output['wavelength'].data
        dwave = np.gradient(cwave)
        # Trim to requested wavelength range.
        waves = get_slice(cwave)
        if waves is None:
            # Skip any cameras outside the requested wavelength range.
            continue
        cwave = cwave[waves]
        dwave = dwave[waves]
        nsky = output['num_sky_electrons'][waves, fiber] / dwave
        nsrc = output['num_source_electrons'][waves, fiber] / dwave
        ndark = output['num_dark_electrons'][waves, fiber] / dwave
        read_noise = (
            output['read_noise_electrons'][waves, fiber] / np.sqrt(dwave))
        total_noise = (
            np.sqrt(output['variance_electrons'][waves, fiber] / dwave))
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

    if camera_output:
        # This kludge is because the label arg to fill_between() does not
        # propagate to legend() in matplotlib < 1.5.
        line1, = ax3.plot([], [], 'k-')
        dark_fill = Rectangle((0, 0), 1, 1, fc='k', alpha=0.2)
        ax3.legend(
            (sky_fill, src_fill, dark_fill, line1, line2),
            ('Sky detected', 'Source detected', 'Dark current',
             'RMS total noise', 'RMS read noise'),
            loc='best', fancybox=True, framealpha=0.5, ncol=5,
            fontsize=label_size)

        ax3.set_ylim(min_electrons, 2e2 * min_electrons)
        ax3.set_yscale('log')
        ax3.set_ylabel('Mean electrons / {0}'.format(waveunit))
        ax3.set_xlim(wave[0], wave[-1])
        ax3.set_xlabel('Wavelength [{0}]'.format(waveunit))

    # Remove x-axis ticks on the upper panels.
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
