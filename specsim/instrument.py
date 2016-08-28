# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an instrument response for spectroscopic simulations.

An instrument model is usually initialized from a configuration used to create
a simulator and then accessible via its ``instrument`` attribute, for example:

    >>> import specsim.simulator
    >>> simulator = specsim.simulator.Simulator('test')
    >>> print(np.round(simulator.instrument.fiber_diameter, 1))
    107.0 um

See :doc:`/api` for examples of changing model parameters defined in the
configuration. No attributes can be changed after a simulator has
been created.  File a github issue if you would like to change this.

An :class:`Instrument` includes one or more
:class:`Cameras <specsim.camera.Camera>`.
"""
from __future__ import print_function, division

import numpy as np
import scipy.interpolate
import scipy.integrate

import astropy.constants
import astropy.units as u

import specsim.camera


class Instrument(object):
    """Model the instrument response of a fiber spectrograph.

    A spectrograph can have multiple :mod:`cameras <specsim.camera>` with
    different wavelength coverages.

    No instrument attributes can be changed after an instrument has been
    created. File a github issue if you would like to change this.

    Parameters
    ----------
    name : str
        Descriptive name of this instrument.
    wavelength : nastropy.units.Quantity
        Array of wavelength bin centers where the instrument response is
        calculated, with units.
    fiber_acceptance_dict : dict
        Dictionary of fiber acceptance fractions tabulated for different
        source models, with keys corresponding to source model names.
    cameras : list
        List of :class:`specsim.camera.Camera` instances representing the
        camera(s) of this instrument.
    primary_mirror_diameter : astropy.units.Quantity
        Diameter of the primary mirror, with units.
    obscuration_diameter : astropy.units.Quantity
        Diameter of a central obscuration of the primary mirror, with units.
    support_width : astropy.units.Quantity
        Width of the obscuring supports, with units.
    fiber_diameter : astropy.units.Quantity
        Physical diameter of the simulated fibers, with units of length.
        Converted to an on-sky diameter using the plate scale.
    field_radius : astropy.units.Quantity
        Maximum radius of the field of view in length units measured at
        the focal plane. Converted to an angular field of view using the
        plate scale.
    radial_scale : callable
        Callable function that returns the plate scale in the radial
        (meridional) direction (with appropriate units) as a function of
        focal-plane distance (with length units) from the boresight.
    azimuthal_scale : callable
        Callable function that returns the plate scale in the azimuthal
        (sagittal) direction (with appropriate units) as a function of
        focal-plane distance (with length units) from the boresight.
    """
    def __init__(self, name, wavelength, fiber_acceptance_dict, cameras,
                 primary_mirror_diameter, obscuration_diameter, support_width,
                 fiber_diameter, field_radius, radial_scale, azimuthal_scale):
        self.name = name
        self._wavelength = wavelength
        self.fiber_acceptance_dict = fiber_acceptance_dict
        self.cameras = cameras
        self.primary_mirror_diameter = primary_mirror_diameter
        self.obscuration_diameter = obscuration_diameter
        self.support_width = support_width
        self.fiber_diameter = fiber_diameter
        self.field_radius = field_radius
        self.radial_scale = radial_scale
        self.azimuthal_scale = azimuthal_scale

        self.source_types = self.fiber_acceptance_dict.keys()

        # Calculate the effective area of the primary mirror.
        D = self.primary_mirror_diameter
        obs = self.obscuration_diameter
        support_area = 0.5*(D - obs) * self.support_width
        self.effective_area = (
            np.pi * ((0.5 * D) ** 2 - (0.5 * obs) ** 2) - 4 * support_area)

        # Tabulate the mapping between focal plane radius and boresight
        # opening angle by integrating the radial plate scale.
        # Use mm and radians as the canonical units.
        self._radius_unit, self._angle_unit = u.mm, u.rad
        radius = np.linspace(
            0., self.field_radius.to(self._radius_unit).value, 1000)
        dradius_dangle = self.radial_scale(radius * self._radius_unit).to(
            self._radius_unit / self._angle_unit).value
        angle = scipy.integrate.cumtrapz(
            1. / dradius_dangle, radius, initial=0.)

        # Record the maximum field angle corresponding to our field radius.
        self.field_angle = angle[-1] * self._angle_unit

        # Build dimensionless linear interpolating functions of the
        # radius <-> angle map using the canonical units.
        self._radius_to_angle = scipy.interpolate.interp1d(
            radius, angle, kind='linear', copy=True, bounds_error=True)
        self._angle_to_radius = scipy.interpolate.interp1d(
            angle, radius, kind='linear', copy=True, bounds_error=True)

        # Calculate the energy per photon at each wavelength.
        hc = astropy.constants.h * astropy.constants.c
        energy_per_photon = (hc / self._wavelength).to(u.erg)

        # Calculate the rate of photons incident on the focal plane per
        # wavelength bin per unit spectral flux density. The fiber acceptance
        # fraction is not included in this calculation.
        wavelength_bin_size = np.gradient(self._wavelength)
        self.photons_per_bin = (
            self.effective_area * wavelength_bin_size / energy_per_photon
            ).to((u.cm**2 * u.Angstrom) / u.erg)

        wave_mid = []
        for i, camera in enumerate(self.cameras):
            wave_min, wave_max = camera.wavelength_min, camera.wavelength_max
            wave_mid.append(0.5 * (wave_min + wave_max))
            if i == 0:
                self.wavelength_min = wave_min
                self.wavelength_max = wave_max
            else:
                self.wavelength_min = min(self.wavelength_min, wave_min)
                self.wavelength_max = max(self.wavelength_max, wave_max)

        # Sort cameras in order of increasing wavelength.
        self.cameras = [x for (y, x) in sorted(zip(wave_mid, self.cameras))]


    def field_radius_to_angle(self, radius):
        """Convert focal plane radius to an angle relative to the boresight.

        The mapping is derived from the radial (meridional) plate scale
        function :math:`dr/d\\theta(r)` via the integral:

        .. math::

            \\theta(r) = \int_0^{r} \\frac{dr}{dr/d\\theta(r')}\, dr'

        The input values must be within the field of view.
        Use :meth:`field_angle_to_radius` for the inverse transform.

        Parameters
        ----------
        radius : astropy.units.Quantity
            One or more radius values where the angle should be calculated.
            Values must be between 0 and ``field radius``.

        Returns
        -------
        astropy.units.Quantity
            Opening angle(s) relative to the boresight corresponding to
            the input radius value(s).

        Raises
        ------
        ValueError
            One or more input values are outside the allowed range.
        """
        return self._radius_to_angle(
            radius.to(self._radius_unit)) * self._angle_unit


    def field_angle_to_radius(self, angle):
        """Convert focal plane radius to an angle relative to the boresight.

        The mapping :math:`r(\\theta)` is calculated by numerically inverting
        the function :math:`\\theta(r)`.

        The input values must be within the field of view.
        Use :meth:`field_radius_to_angle` for the inverse transform.

        Parameters
        ----------
        angle : astropy.units.Quantity
            One or more angle values where the radius should be calculated.
            Values must be between 0 and ``field_angle``.

        Returns
        -------
        astropy.units.Quantity
            Radial coordinate(s) in the focal plane corresponding to the
            input angle value(s).

        Raises
        ------
        ValueError
            One or more input values are outside the allowed range.
        """
        return self._angle_to_radius(
            angle.to(self._angle_unit)) * self._radius_unit


    def plot_field_distortion(self):
        """Plot focal plane distortions over the field of view.

        Requires that the matplotlib package is installed.
        """
        import matplotlib.pyplot as plt

        # Tabulate the field radius - angle mapping.
        radius = np.linspace(0., self.field_radius.to(u.mm).value, 500) * u.mm
        angle = self.field_radius_to_angle(radius).to(u.deg)

        # Calculate the r**2 weighted mean inverse radial scale by minimizing
        # angle - mean_inv_radial_scale * radius with respect to
        # mean_inv_radial_scale.
        mean_inv_radial_scale = (
            np.sum(radius ** 3 * angle) / np.sum(radius ** 4))
        mean_radial_scale = (1. / mean_inv_radial_scale).to(u.um / u.arcsec)

        # Calculate the angular distortion relative to the mean radial scale.
        distortion = (angle - radius * mean_inv_radial_scale).to(u.arcsec)

        # Eliminate round off error so that the zero distortion case is
        # correctly recovered.
        distortion = np.round(distortion, decimals=5)

        # Calculate the fiber area as a function of radius.
        radial_size = (
            0.5 * self.fiber_diameter / self.radial_scale(radius))
        azimuthal_size = (
            0.5 * self.fiber_diameter / self.azimuthal_scale(radius))
        fiber_area = (np.pi * radial_size * azimuthal_size).to(u.arcsec ** 2)

        # Calculate the r**2 weighted mean fiber area.
        mean_fiber_area = np.sum(radius ** 2 * fiber_area) / np.sum(radius ** 2)

        # Calculate the dimensionless fiber area ratio.
        fiber_area_ratio = (fiber_area / mean_fiber_area).si.value

        # Calculate the dimensionless ratio of azimuthal / radial plate scales
        # which is the ratio of the on-sky radial / azimuthal extends.
        shape_ratio = (self.azimuthal_scale(radius) /
                       self.radial_scale(radius)).si.value

        # Make the plots.
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 8))

        ax1.plot(angle, distortion, 'b-', lw=2)
        ax1.set_ylabel('Field angle distortion [arcsec]', fontsize='large')
        ax1.set_xlim(0., self.field_angle.to(u.deg).value)
        ax1.grid()

        ax1.axhline(0., color='r')
        xy = 0.5 * self.field_angle.to(u.deg).value, 0.
        label = '{0:.1f}'.format(mean_radial_scale)
        ax1.annotate(label, xy, xy, color='r', horizontalalignment='center',
                     verticalalignment='bottom', fontsize='large')

        ax2.plot(angle, fiber_area_ratio, 'b', lw=2, label='Area ratio')
        ax2.plot(angle, shape_ratio, 'k', lw=2, ls='--',
                 label='Radial/azimuthal')
        ax2.set_ylabel('Fiber sky area and shape ratios', fontsize='large')
        ax2.grid()
        ax2.legend(loc='upper right')

        ax2.axhline(1., color='r')
        xy = 0.5 * self.field_angle.to(u.deg).value, 1.
        label = '{0:.3f}'.format(mean_fiber_area)
        ax2.annotate(label, xy, xy, color='r', horizontalalignment='center',
                     verticalalignment='bottom', fontsize='large')

        ax2.set_xlabel('Field angle [deg]', fontsize='large')
        plt.subplots_adjust(
            left=0.10, right=0.98, bottom=0.07, top=0.97, hspace=0.05)


    def get_fiber_acceptance(self, source):
        """Get the tabulated fiber acceptance function for the specified source.

        Parameters
        ----------
        source : specsim.source.Source
            The source whose fiber acceptance should be returned.

        Returns
        -------
        numpy.ndarray
            Array of fiber acceptance values in the range 0-1, tabulated at
            at each :attr:`wavelength`.
        """
        if source.type_name not in self.source_types:
            raise ValueError(
                "Invalid source type '{0}'. Pick one of {1}."
                .format(source.type_name, self.source_types))
        return self.fiber_acceptance_dict[source.type_name]


    def plot(self, flux=1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom),
             exposure_time=1000 * u.s, cmap='nipy_spectral'):
        """Plot a summary of this instrument's model.

        Requires that the matplotlib package is installed.

        Parameters
        ----------
        flux : astropy.units.Quantity
            Constant source flux to use for displaying the instrument response.
        exposure_time : astropy.units.Quantity
            Exposure time to use for displaying the instrument response.
        cmap : str or matplotlib.colors.Colormap
            Matplotlib colormap name or instance to use for displaying the
            instrument response.  Colors are selected for each camera
            according to its central wavelength, so a spectral color map
            will give reasonably intuitive results.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 8))
        ax1_rhs = ax1.twinx()
        ax2_rhs = ax2.twinx()
        cmap = cm.get_cmap(cmap)

        wave = self._wavelength.value
        wave_unit = self._wavelength.unit
        dwave = np.gradient(wave)

        for source_type in self.source_types:
            # Plot fiber acceptance fractions without labels.
            ax1.plot(wave, self.fiber_acceptance_dict[source_type], 'k--')
        for camera in self.cameras:
            cwave = camera._wavelength

            # Use an approximate spectral color for each band.
            mid_wave = 0.5 * (camera.wavelength_min + camera.wavelength_max)
            color = cmap(
                (mid_wave - self.wavelength_min) /
                (self.wavelength_max - self.wavelength_min))

            # Calculate number of photons with perfect fiber acceptance.
            nphot = (flux * self.photons_per_bin * exposure_time *
                     camera.throughput / dwave)
            dark_noise = np.sqrt(
                (camera.dark_current_per_bin * exposure_time).value)
            total_noise = np.sqrt(
                dark_noise ** 2 + camera.read_noise_per_bin.value ** 2)

            ax1.plot(cwave, camera.throughput, ls='-', color=color)

            ax1_rhs.plot(cwave, nphot.value, ls=':', color=color)
            ax1_rhs.fill_between(
                cwave, total_noise / dwave, lw=0, color=color, alpha=0.2)
            ax1_rhs.fill_between(
                cwave, dark_noise / dwave, lw=0, color=color, alpha=0.2)
            ax1_rhs.plot(cwave, total_noise / dwave, ls='-.', color=color)

            ax2.plot(
                cwave, camera.rms_resolution.to(wave_unit).value,
                ls='-', color=color)
            ax2.plot(
                cwave, camera.row_size.to(wave_unit / u.pixel).value,
                ls='--', color=color)

            ax2_rhs.plot(
                cwave, camera.neff_spatial.to(u.pixel), ls=':', color=color)

        ax1.plot([], [], 'k--', label='Fiber Acceptance')
        ax1.plot([], [], 'k-', label='Camera Throughput')
        ax1.plot([], [], 'k:', label='{0}'.format(flux))
        ax1.plot([], [], 'k-.', label='Dark + Readout Noise')
        ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        ax2.plot([], [], 'k-', label='RMS Resolution')
        ax2.plot([], [], 'k--', label='Row Size')
        ax2.plot([], [], 'k:', label='Column Size')
        ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)

        ax1.set_ylim(0., None)
        ax1.set_ylabel('Fiber, Camera Throughput')
        ax1_rhs.set_ylim(0., None)
        ax1_rhs.set_ylabel(
            'Photons, Electrons / Exposure / {0}'.format(wave_unit))
        ax2.set_ylim(0., None)
        ax2.set_ylabel('RMS Resolution, Row Size [{0}]'.format(wave_unit))
        ax2_rhs.set_ylim(0., None)
        ax2_rhs.set_ylabel('Effective Column Size [pixels]')
        ax2.set_xlabel('Wavelength [{0}]'.format(wave_unit))
        ax2.set_xlim(wave[0], wave[-1])


def initialize(config):
    """Initialize the instrument model from configuration parameters.

    This method is responsible for creating a new :class:`Instrument` as
    well as the :class:`Cameras <specsim.camera.Camera>` it includes.

    Parameters
    ----------
    config : :class:`specsim.config.Configuration`
        The configuration parameters to use.

    Returns
    -------
    Instrument
        An initialized instrument model including one or more
        :class:`cameras <specsim.camera.Camera>`.
    """
    name = config.instrument.name
    cameras = config.instrument.cameras
    camera_names = cameras.keys()
    initialized_cameras = []
    for camera_name in camera_names:
        camera = getattr(cameras, camera_name)
        ccd = config.load_table(
            camera.ccd, ['row_size', 'fwhm_resolution', 'neff_spatial'])
        throughput = config.load_table(camera.throughput, 'throughput')
        constants = config.get_constants(camera,
            ['read_noise', 'dark_current', 'gain', 'num_sigmas_clip',
             'output_pixel_size'])
        initialized_cameras.append(specsim.camera.Camera(
            camera_name, config.wavelength, throughput,
            ccd['row_size'], ccd['fwhm_resolution'],
            ccd['neff_spatial'], constants['read_noise'],
            constants['dark_current'], constants['gain'],
            constants['num_sigmas_clip'], constants['output_pixel_size']))

    constants = config.get_constants(
        config.instrument,
        ['primary_mirror_diameter', 'obscuration_diameter',
         'support_width', 'fiber_diameter', 'field_radius'])

    try:
        # Try to read a tabulated plate scale first.
        plate_scale = config.load_table(
            config.instrument.plate_scale,
            ['radius', 'radial_scale', 'azimuthal_scale'], interpolate=False)
        r_vec = plate_scale['radius']
        sr_vec = plate_scale['radial_scale']
        sa_vec = plate_scale['azimuthal_scale']
        # Build dimensionless linear interpolators for the radial and azimuthal
        # scales using the native units from the tabulated data.
        sr_interpolate = scipy.interpolate.interp1d(
            r_vec.value, sr_vec.value, kind='linear', copy=True)
        sa_interpolate = scipy.interpolate.interp1d(
            r_vec.value, sa_vec.value, kind='linear', copy=True)
        # Wrap interpolators in lambdas that take care of units.
        radial_scale = lambda r: (
            sr_interpolate(r.to(r_vec.unit).value) * sr_vec.unit)
        azimuthal_scale = lambda r: (
            sa_interpolate(r.to(r_vec.unit).value) * sa_vec.unit)
    except AttributeError:
        # Fall back to a constant value.
        plate_scale_constant = config.get_constants(
            config.instrument.plate_scale, ['value'])
        value = plate_scale_constant['value']
        # Create lambdas that return the constant plate scale with units.
        # Use np.ones_like to ensure correct broadcasting.
        radial_scale = lambda r: value * np.ones_like(r.value)
        azimuthal_scale = lambda r: value * np.ones_like(r.value)

    fiber_acceptance_dict = config.load_table(
        config.instrument.fiberloss, 'fiber_acceptance', as_dict=True)

    instrument = Instrument(
        name, config.wavelength, fiber_acceptance_dict, initialized_cameras,
        constants['primary_mirror_diameter'], constants['obscuration_diameter'],
        constants['support_width'], constants['fiber_diameter'],
        constants['field_radius'], radial_scale, azimuthal_scale)

    if config.verbose:
        # Print some derived quantities.
        print('Telescope effective area: {0:.3f}'
              .format(instrument.effective_area))
        print('Field of view diameter: {0:.1f} = {1:.2f}.'
              .format(2 * instrument.field_radius.to(u.mm),
                      2 * instrument.field_angle.to(u.deg)))
        print('Source types: {0}.'.format(instrument.source_types))

    return instrument
