# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an instrument response for spectroscopic simulations.

An instrument model is usually initialized from a configuration used to create
a simulator and then accessible via its ``instrument`` attribute, for example:

    >>> import specsim.simulator
    >>> simulator = specsim.simulator.Simulator('test')  # doctest: +IGNORE_OUTPUT
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
import os.path
import scipy.interpolate
import scipy.integrate

import astropy.constants
import astropy.units as u

import specsim.camera
import specsim.fastfiberacceptance
import specsim.config

class Instrument(object):
    """Model the instrument response of a fiber spectrograph.

    A spectrograph can have multiple :mod:`cameras <specsim.camera>` with
    different wavelength coverages. Objects representing each camera are
    contained in a list accessible from our ``cameras`` attribute, which will
    be in order of increasing effective wavelength.

    No instrument attributes can be changed after an instrument has been
    created. Create a github issue if you would like to change this.

    Parameters
    ----------
    name : str
        Descriptive name of this instrument.
    wavelength : astropy.units.Quantity
        Array of wavelength bin centers where the instrument response is
        calculated, with units.
    fiberloss_method : str
        Must be "table" or "galsim" or "fastsim".  Specifies how fiber acceptance fractions
        will be loaded or calculated.
    fiber_acceptance_dict : dict or None
        Dictionary of fiber acceptance fractions tabulated for different
        source models, with keys corresponding to source model names.
        Ignored when fiberloss_method is "galsim".
    fast_fiber_acceptance : specsim.fastfiberacceptance.FastFiberAcceptance or None
        Initialized instance to use when fiberloss_method is "fastsim".
        Ignored for other values of fiberloss_method.
    fiberloss_num_wlen : int
        Number of wavelengths where the fiberloss fraction should be tabulated
        for interpolation.  Ignored when fiberloss_method is not "galsim".
    fiberloss_num_pixels : int
        Number of pixels used to subdivide the fiber diameter for
        numerical convolution and integration calculations.
        Ignored when fiberloss_method  is not "galsim".
    blur_function : callable
        Function of field angle and wavelength that returns the corresponding
        RMS blur in length units (e.g., microns).
    offset_function : callable
        Function of focal-plane position (x,y) in angular units and wavelength
        that returns the corresponding radial centroid offset in length
        units (e.g., microns).
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
    def __init__(self, name, wavelength, fiberloss_method,
                 fiber_acceptance_dict, fast_fiber_acceptance, fiberloss_num_wlen,
                 fiberloss_num_pixels, blur_function, offset_function, cameras,
                 primary_mirror_diameter, obscuration_diameter, support_width,
                 fiber_diameter, field_radius, radial_scale, azimuthal_scale):
        self.name = name
        self._wavelength = wavelength
        self.fiber_acceptance_dict = fiber_acceptance_dict
        self.fast_fiber_acceptance = fast_fiber_acceptance
        # Both fiber_acceptance_dict and fast_fiber_acceptance must be initialized
        # before assigning to fiberloss_method (since its setter checks their values).
        self.fiberloss_method = fiberloss_method
        self.fiberloss_num_wlen = fiberloss_num_wlen
        self.fiberloss_num_pixels = fiberloss_num_pixels
        self._blur_function = blur_function
        self._offset_function = offset_function
        self.cameras = cameras
        self.primary_mirror_diameter = primary_mirror_diameter
        self.obscuration_diameter = obscuration_diameter
        self.support_width = support_width
        self.fiber_diameter = fiber_diameter
        self.field_radius = field_radius
        self.radial_scale = radial_scale
        self.azimuthal_scale = azimuthal_scale

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


    @property
    def fiberloss_method(self):
        """The current method used to calculate fiber acceptance fractions.
        """
        return self._fiberloss_method


    @fiberloss_method.setter
    def fiberloss_method(self, fiberloss_method):
        """Set the method used to calculate fiber acceptance fractions.

        Must be one of "table" or "galsim" or "fastsim".
        """
        if fiberloss_method not in ('table', 'galsim', 'fastsim' ):
            raise ValueError(
                'fiberloss_method must be "table" or "galsim" or "fastsim".')
        if fiberloss_method == 'table' and self.fiber_acceptance_dict is None:
            raise ValueError('Missing required instrument.fiberloss.table.')
        if fiberloss_method == 'fastsim' and self.fast_fiber_acceptance is None:
            raise ValueError(
                'Missing required instrument.fiberloss.fast_fiber_acceptance_path.')
        if fiberloss_method == 'galsim':
            try:
                import galsim
            except ImportError:
                raise ValueError('The galsim package is not installed.')
        self._fiberloss_method = fiberloss_method


    def field_radius_to_angle(self, radius):
        """Convert focal plane radius to an angle relative to the boresight.

        The mapping is derived from the radial (meridional) plate scale
        function :math:`dr/d\\theta(r)` via the integral:

        .. math::

            \\theta(r) = \\int_0^{r} \\frac{dr}{dr/d\\theta(r')}\\, dr'

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


    def get_blur_rms(self, wavelength, angle):
        """Get the instrument PSF blur at the specified field angle.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength where the blur should be calculated.
        angle : astropy.units.Quantity
            Angular separation from the field center.

        Returns
        -------
        astropy.units.Quantity
            RMS blur of the instrument at this wavelength and field radius
            in length units.
        """
        return self._blur_function(angle, wavelength)


    def get_centroid_offset(self, angle_x, angle_y, wavelength):
        """Get the instrument centroid offset at the specified field angles.

        This method does not make any assumptions about how the x and y
        axes are defined, as long as (0, 0) is the field center.

        Note that the focal-plane position is input as angles relative to
        the field center, while the offsets are returned as lengths relative
        to the nominal fiber center.

        Parameters
        ----------
        angle_x : astropy.units.Quantity
            Angular separation from the field center along x.
        angle_y : astropy.units.Quantity
            Angular separation from the field center along y.
        wavelength : astropy.units.Quantity
            Wavelength where the blur should be calculated.

        Returns
        -------
        tuple
            Tuple (dx, dy) of astropy quantities giving the spot centroid
            offset components at this wavelength and position in the focal
            plane.  Offsets are given in length units, e.g., microns.
        """
        return self._offset_function(angle_x, angle_y, wavelength)


    def get_focal_plane_optics(self, focal_x, focal_y, wlen_grid):
        """Calculate the optical parameters at a set of focal-plane positions.

        Uses :meth:`get_centroid_offset`, :meth:`get_blur_rms`, and
        :meth:`field_radius_to_angle` to calculate the optics at each focal
        plane location.

        This method does not make any assumptions about how the x and y
        axes are defined, as long as (0, 0) is the field center.  However
        radial symmetry is broken by the (dx, dy) offsets calculated by
        :meth:`get_centroid_offset`.

        Note that units are required for the input arrays and included with
        the returned arrays.

        Parameters
        ----------
        focal_x : :class:`astropy.units.Quantity`
            1D array of X coordinates in the focal plane relative to the
            boresight, with length units.
        focal_y : :class:`astropy.units.Quantity`
            1D array of Y coordinates in the focal plane relative to the
            boresight, with length units.
        wlen_grid : :class:`astropy.units.Quantity`
            1D array of wavelengths where parameters should be tabulated,
            with length units.

        Returns
        -------
        tuple
            Tuple of arrays scale, blur, offset with shapes (N,2), (N,M) and
            (N,M,2) where N is the size of the 1D input (x,y) arrays, M is
            the size of the input wavelength grid, and axes of length 2
            correspond to radial and azimuthal axes (not the input x,y!).
            All output arrays have units.
        """
        # Check for valid units on the input arrays.
        try:
            focal_x_mm = focal_x.to(u.mm).value
            focal_y_mm = focal_y.to(u.mm).value
            wlen_grid_ang = wlen_grid.to(u.Angstrom).value
        except astropy.units.UnitConversionError:
            raise ValueError('Input arrays have invalid units.')
        except AttributeError:
            raise ValueError('Input arrays are missing required units.')

        # Check for expected input array shapes.
        if len(focal_x_mm.shape) != 1 or len(wlen_grid_ang.shape) != 1:
            raise ValueError('Input arrays must be 1D.')
        if focal_x_mm.shape != focal_y_mm.shape:
            raise ValueError('Input (x,y) arrays have different shapes.')

        # Allocate output arrays.
        n_xy = len(focal_x_mm)
        n_wlen = len(wlen_grid_ang)
        scale = np.empty((n_xy, 2))
        blur = np.empty((n_xy, n_wlen))
        offset = np.empty((n_xy, n_wlen, 2))

        # Convert x, y offsets in length units to field angles.
        focal_r = np.sqrt(focal_x**2+focal_y**2)
        angle_r = self.field_radius_to_angle(focal_r)
        angle_x = np.zeros(focal_x.shape) * angle_r.unit
        angle_y = np.zeros(focal_y.shape) * angle_r.unit
        positive_radius = focal_r>0
        angle_x[positive_radius] = (
            angle_r[positive_radius] / focal_r[positive_radius]
            ) * focal_x[positive_radius]
        angle_y[positive_radius] = (
            angle_r[positive_radius] / focal_r[positive_radius]
            ) * focal_y[positive_radius]

        # Calculate the radial and azimuthal plate scales at each location.
        scale[:, 0] = self.radial_scale(focal_r).to(u.um / u.arcsec).value
        scale[:, 1] = self.azimuthal_scale(focal_r).to(u.um / u.arcsec).value

        # Calculate the transformations between polar and Cartesian coordinates.
        phi = np.arctan2(focal_y_mm, focal_x_mm)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Lookup the instrument blur and centroid offset at each
        # wavelength for this focal-plane position.
        for i, wlen in enumerate(wlen_grid):
            # Lookup the RMS blurs in focal-plane microns.
            blur[:, i] = self.get_blur_rms(wlen, angle_r).to(u.um).value
            # Lookup the radial centroid offsets in focal-plane microns.
            dx, dy = self.get_centroid_offset(angle_x, angle_y, wlen)
            dx_um = dx.to(u.um).value
            dy_um = dy.to(u.um).value
            # Rotate to polar coordinates.
            offset[:, i, 0] =  cos_phi * dx_um + sin_phi * dy_um
            offset[:, i, 1] =  -sin_phi * dx_um + cos_phi * dy_um

        return scale * (u.um / u.arcsec), blur * u.um, offset * u.um


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

        if self.fiber_acceptance_dict:
            for source_type in self.fiber_acceptance_dict:
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


def initialize(config, camera_output=True):
    """Initialize the instrument model from configuration parameters.

    This method is responsible for creating a new :class:`Instrument` as
    well as the :class:`Cameras <specsim.camera.Camera>` it includes.

    Parameters
    ----------
    config : :class:`specsim.config.Configuration`
        The configuration parameters to use.
    camera_output : bool
        Initialize support for resolution convolution and downsampling for
        each camera when True.

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
            constants['num_sigmas_clip'], constants['output_pixel_size'],
            allow_convolution=camera_output))

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

    # Initialize for both fiberloss methods so that method can be changed
    # at run time.
    fiberloss_method = config.instrument.fiberloss.method
    fiberloss_num_wlen = config.instrument.fiberloss.num_wlen
    fiberloss_num_pixels = config.instrument.fiberloss.num_pixels
    if hasattr(config.instrument.fiberloss, 'table'):
        fiber_acceptance_dict = config.load_table(
            config.instrument.fiberloss, 'fiber_acceptance', as_dict=True)
    else:
        fiber_acceptance_dict = None
    if hasattr(config.instrument.fiberloss, 'fast_fiber_acceptance_path'):
        filename = os.path.join(
            config.abs_base_path,
            config.instrument.fiberloss.fast_fiber_acceptance_path)
        if not os.path.isfile(filename) :
            raise RuntimeError(
            'Cannot find file {}. May need to update desimodel svn ?'
            .format(filename))
        fast_fiber_acceptance = specsim.fastfiberacceptance.FastFiberAcceptance(
            filename)
    else:
        fast_fiber_acceptance = None

    blur_value = getattr(config.instrument.blur, 'value', None)
    if blur_value:
        blur_value = specsim.config.parse_quantity(blur_value, u.micron)
        blur_function = lambda angle, wlen: blur_value
    else:
        blur_function = config.load_table2d(
            config.instrument.blur, 'wavelength', 'r=')

    offset_value = getattr(config.instrument.offset, 'value', None)
    if offset_value:
        offset_value = specsim.config.parse_quantity(offset_value, u.micron)
        offset_function = (
            lambda angle_x, angle_y, wlen: (offset_value, 0 * u.um))
    else:
        # Build an interpolator in (r, wlen) of radial chromatic offsets.
        radial_offset_function = config.load_table2d(
            config.instrument.offset, 'wavelength', 'r=')
        # Look for an optional file of achromatic offsets.
        # Static achromatic term, correlated in focal plane
        if hasattr(config.instrument.offset, 'static'):
            # Build an interpolator in (x, y).
            interpolators = config.load_fits2d(
                config.instrument.offset.static, xy_unit=u.deg,
                dx='XOFFSET', dy='YOFFSET')
            static_fx = interpolators['dx']
            static_fy = interpolators['dy']
        else:
            static_fx  = None
            static_fy  = None
        # Random uncorrelated achromatic term
        if hasattr(config.instrument.offset, 'sigma1d'):
            if hasattr(config.instrument.offset, 'seed'):
                offset_gen = np.random.RandomState(config.instrument.offset.seed)
            else:
                offset_gen = np.random.RandomState()
            sigma1d=specsim.config.parse_quantity(config.instrument.offset.sigma1d)
            random_fx = lambda angle_x, angle_y: offset_gen.normal(size=angle_x.shape) * sigma1d
            random_fy = lambda angle_x, angle_y: offset_gen.normal(size=angle_x.shape) * sigma1d
        else :
            random_fx = None
            random_fy = None
        # Combine the interpolators into a function of (x, y, wlen) that
        # returns (dx, dy).  Use default parameter values to capture the
        # necessary state in the inner function's closure.
        def offset_function(angle_x, angle_y, wlen,
                            fr=radial_offset_function,
                            static_fx=static_fx,
                            static_fy=static_fy,
                            random_fx=random_fx,
                            random_fy=random_fy) :
            angle_r = np.sqrt(angle_x ** 2 + angle_y ** 2)
            dr = fr(angle_r, wlen)
            # Special handling of the origin.
            not_at_origin = (angle_r > 0.)
            ux = np.ones(shape=dr.shape, dtype=float)
            uy = np.ones(shape=dr.shape, dtype=float)
            ux[not_at_origin] = angle_x / angle_r
            uy[not_at_origin] = angle_y / angle_r
            dx = dr * ux
            dy = dr * uy
            # Add interpolated offsets if any.
            if static_fx is not None : dx += static_fx(angle_x, angle_y)
            if static_fy is not None : dy += static_fy(angle_x, angle_y)
            if random_fx is not None : dx += random_fx(angle_x, angle_y)
            if random_fy is not None : dy += random_fy(angle_x, angle_y)
            return dx , dy

    instrument = Instrument(
        name, config.wavelength, fiberloss_method, fiber_acceptance_dict,
        fast_fiber_acceptance, fiberloss_num_wlen, fiberloss_num_pixels,
        blur_function, offset_function, initialized_cameras,
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
        if fiber_acceptance_dict is not None:
            print('Fiberloss source types: {0}.'
                  .format(instrument.fiber_acceptance_dict.keys()))

    return instrument
