# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model an astronomical source for spectroscopic simulations.

An source model is usually initialized from a configuration used to create
a simulator and then accessible via its ``source`` attribute, for example:

    >>> import specsim.simulator
    >>> simulator = specsim.simulator.Simulator('test')  # doctest: +IGNORE_OUTPUT
    >>> print(simulator.source.name)
    Constant flux density test source

After initialization, all aspects of a source can be modified at runtime.
"""
from __future__ import print_function, division

import numpy as np

import scipy.interpolate

import astropy.units as u

import speclite.filters

import specsim.config


class Source(object):
    """Source model used for simulation.

    A source is defined on both an input and output wavelength grid. The
    input grid represents the best knowledge of the source over the widest
    possible wavelength range, to allow for redshift transforms and filter
    calculations via :meth:`get_flux_out`. The output grid is determined by the
    simulation and represents observed wavelengths in the instrument.

    All parameters except for ``wavelength_out`` can be modified using
    :meth:`update_in` and :meth:`update_out`. A simulation uses only the
    attribute :attr:`flux_out` for its calculations.

    The simulation needs to locate a source in the focal plane.  This is
    either done by specifying (x,y) coordinates in the focal plane, or else
    by specifying the sky position of the source and calculating its
    focal plane coordinates from the observing time, pointing and atmospheric
    conditions.

    Parameters
    ----------
    name : str
        Brief descriptive name of this model.
    type_name : str
        Name of the instrument fiber acceptance model that should be used
        to simulate this source.
    wavelength_out : astropy.units.Quantity
        Array of increasing output wavelengths with units.
    wavelength_in : astropy.units.Quantity
        Array of increasing input wavelengths with units.
    flux_in : astropy.units.Quantity
        Array of input flux values tabulated at wavelength_in.
    disk_fraction : float
        Fraction of flux in disk (Sersic n=1) component.  Must be between 0
        and 1, and sum of disk_fraction and bulge_fraction must be <= 1.
        If sum is < 1, the remainder is point like.
    bulge_fraction : float
        Fraction of flux in bulge (Sersic n=4) component.  Must be between 0
        and 1, and sum of disk_fraction and bulge_fraction must be <= 1.
        If sum is < 1, the remainder is point like.
    disk_shape : Profile
        Transverse profile of disk component with Sersic n=1. Ignored when
        disk_fraction is 0.
    bulge_shape : Profile
        Transverse profile of bulge component with Sersic n=4. Ignored when
        disk_fraction is 1.
    focal_xy : astropy.units.Quantity or None
        Astropy quantity of shape (nfiber, 2) giving the focal plane coordinates
        where this source is observed.  When None, the focal plane position is
        calculated from the sky_position and observing conditions.
    sky_position : astropy.coordinates.SkyCoord or None
        Location of this source in the sky. A source will not be visible
        unless its location is within the instrument field of view. Used to
        determine the location of this source on the focal plane, using
        the observing time, pointing and atmospheric conditions. Ignored
        when focal_xy is not None.
    z_in : float or None
        Redshift of (wavelength_in, flux_in) to assume for redshift transforms.
        Ignored unless z_out is set and must be set when z_out is set.
    z_out : float or None
        When this parameter is set, (:attr:`wavelength_in`, :attr:`flux_in`)
        are redshifted from z_in to this value to obtain :attr:`flux_out`.
    filter_name : str or None
        Name of the `speclite filter response
        <http://speclite.readthedocs.io/en/stable/filters.html>`__ to use
        for normalizing :attr:`flux_out`. Ignored when ab_magnitude_out is None.
    ab_magnitude_out : float or None
        AB magnitude to use for normalizing :attr:`flux_out`.  Note that any
        redshift transform is applied before normalizing.
    """
    def __init__(self, name, type_name, wavelength_out, wavelength_in, flux_in,
                 disk_fraction, bulge_fraction, disk_shape, bulge_shape,
                 focal_xy, sky_position, z_in=None, z_out=None,
                 filter_name=None, ab_magnitude_out=None):

        wavelength_out = np.asanyarray(wavelength_out)
        if len(wavelength_out.shape) != 1:
            raise ValueError('Expected 1D array for wavelength_out.')
        try:
            converted = wavelength_out.unit.to(u.Angstrom)
        except (AttributeError, u.UnitConversionError):
            raise ValueError('Invalid or missing unit for wavelength_out.')
        self._wavelength_out = wavelength_out.copy()

        self.update_in(name, type_name, wavelength_in, flux_in, z_in)
        self.update_out(z_out, filter_name, ab_magnitude_out)

        if bulge_fraction < 0 or bulge_fraction > 1:
            raise ValueError('Expected bulge_fraction in the range 0-1.')
        if disk_fraction < 0 or disk_fraction > 1:
            raise ValueError('Expected disk_fraction in the range 0-1.')
        if bulge_fraction + disk_fraction > 1:
            raise ValueError(
                'Expected bulge_fraction + disk_fraction <= 1.')
        self.bulge_fraction = bulge_fraction
        self.disk_fraction = disk_fraction
        self.disk_shape = disk_shape
        self.bulge_shape = bulge_shape

        if focal_xy is None and sky_position is None:
            raise ValueError(
                'Either focal_xy or sky_position must be specified.')
        self.focal_xy = focal_xy
        self.sky_position = sky_position


    def update_in(self, name, type_name, wavelength_in, flux_in, z_in=None):
        """Update this source model.

        All parameters have the same meaning as in the
        :class:`constructor <Source>`.  A call to this method must be
        followed by a call to :meth:`update_out`, otherwise an attempt to
        access :attr:`flux_out` will raise a RuntimeError.

        Parameters
        ----------
        name : str
            See :class:`constructor <Source>`.
        type_name : str
            See :class:`constructor <Source>`.
        wavelength_in : astropy.units.Quantity
            See :class:`constructor <Source>`.
        flux_in : astropy.units.Quantity
            See :class:`constructor <Source>`.
        z_in : float or None
            See :class:`constructor <Source>`.
        """
        self._name = name
        self._type_name = type_name

        if z_in is not None:
            z_in = np.float(z_in)
            if z_in <= -1.0:
                raise ValueError('Invalid z_in <= -1.')
        self._z_in = z_in

        # Check for valid shapes.
        wavelength_in = np.asanyarray(wavelength_in)
        flux_in = np.asanyarray(flux_in)
        if len(wavelength_in.shape) != 1:
            raise ValueError('Inputs must be 1D arrays.')
        if len(wavelength_in) != len(flux_in):
            raise ValueError('Input arrays must have same length.')

        # Check for valid units.
        try:
            converted = wavelength_in.unit.to(u.Angstrom)
            converted = flux_in.unit.to(u.erg / (u.s * u.cm **2 * u.Angstrom))
        except (AttributeError, u.UnitConversionError):
            raise ValueError('Inputs have invalid or missing units.')

        self._wavelength_in = wavelength_in.copy()
        self._flux_in = flux_in.copy()

        self._update_out_required = True


    def update_out(self, z_out=None, filter_name=None, ab_magnitude_out=None):
        """Calculate the flux on the output wavelength grid.

        All parameters have the same meaning as in the
        :class:`constructor <Source>`. The result is accessible as
        :attr:`flux_out`.

        Parameters
        ----------
        z_out : float or None
            See :class:`constructor <Source>`. Use :meth:`update_in` to change
            the assumed initial redshift.
        filter_name : str or None
            See :class:`constructor <Source>`.
        ab_magnitude_out : float or None
            See :class:`constructor <Source>`.
        """
        wavelength_unit = self.wavelength_out.unit
        flux_unit = self.flux_in.unit
        wavelength_value = self.wavelength_in.to(wavelength_unit).value.copy()
        flux_value = self.flux_in.value.copy()

        # Appy a redshift transformation, if requested.
        if z_out is not None:
            if self._z_in is None:
                raise RuntimeError(
                    'Cannot redshift unless z_in and z_out are both set.')
            z_ratio = (1. + z_out) / (1. + self._z_in)
            wavelength_value *= z_ratio
            flux_value /= z_ratio

        # Normalize to a specified magnitude, if requested.
        if ab_magnitude_out is not None:
            if filter_name is None:
                raise ValueError(
                    'Must specify filter_name with ab_magnitude_out.')
            filter_response = speclite.filters.load_filter(filter_name)
            ab_magnitude_in = filter_response.get_ab_magnitude(
                flux_value * flux_unit, wavelength_value * wavelength_unit)
            flux_value *= 10 ** (-(ab_magnitude_out - ab_magnitude_in) / 2.5)

        # Interpolate to the output wavelength grid, if necessary.
        if not np.array_equal(wavelength_value, self.wavelength_out.value):
            interpolator = scipy.interpolate.interp1d(
                wavelength_value, flux_value, kind='linear', copy=False)
            flux_out_value = interpolator(self.wavelength_out.value)
        else:
            flux_out_value = flux_value

        self._flux_out = flux_out_value * flux_unit
        self._update_out_required = False


    @property
    def name(self):
        """str: Brief descriptive name of this model.

        Use :meth:`update_in` to change this attribute's value.
        """
        return self._name


    @property
    def type_name(self):
        """str: Name of this source's instrument fiber acceptance model.

        Use :meth:`update_in` to change this attribute's value.
        """
        return self._type_name


    @property
    def wavelength_in(self):
        """astropy.units.Quantity: Array of input wavelengths with units.

        Use :meth:`update_in` to change this attribute's value.
        """
        return self._wavelength_in


    @property
    def flux_in(self):
        """astropy.units.Quantity: Flux values tabulated at wavelength_in.

        Use :meth:`update_in` to change this attribute's value.
        """
        return self._flux_in


    @property
    def wavelength_out(self):
        """astropy.units.Quantity: Array of output wavelengths with units.

        This attribute is read only and fixed by the
        :class:`constructor <Source>`.
        """
        return self._wavelength_out


    @property
    def flux_out(self):
        """astropy.units.Quantity: Flux values tabulated at wavelength_out.

        This attribute is read only and updated by :meth:`update_out`.
        """
        if self._update_out_required:
            raise RuntimeError('update_out() not yet called after update_in().')
        return self._flux_out


class Profile(object):
    """Transverse profile of a single Sersic component of a galaxy.

    If any parameters are strings, they will be converted and validated.

    Parameters
    ----------
    half_light_radius : str or astropy.units.Quantity
        Half-light radius of this component with angular units.
    minor_major_axis_ratio : float
        Ratio of the minor to major ellipse axes q = a/b, which must
        be 0 < q <= 1.
    position_angle : str or astropy.units.Quantity
        Position angle of this component's major axis with angular units.
        Angles are measured counter-clockwise from the +x axis of the focal
        plane coordinate system.
    sersic_index : float
        Sersic index of this component, which must be > 0.
    """
    def __init__(self, half_light_radius, minor_major_axis_ratio,
                 position_angle, sersic_index):
        """Validate and save Sersic component parameters.
        """
        self.half_light_radius = specsim.config.parse_quantity(
            half_light_radius, u.arcsec)
        self.minor_major_axis_ratio = float(minor_major_axis_ratio)
        if self.minor_major_axis_ratio <= 0 or self.minor_major_axis_ratio > 1:
            raise ValueError('Expected minor/major axis ratio in (0,1].')
        self.position_angle = specsim.config.parse_quantity(
            position_angle, u.deg)
        self.sersic_index = float(sersic_index)
        if self.sersic_index <= 0:
            raise ValueError('Expected Sersic index > 0.')


def initialize(config):
    """Initialize the source model from configuration parameters.

    Parameters
    ----------
    config : :class:`specsim.config.Configuration`
        The configuration parameters to use.

    Returns
    -------
    Source
        An initialized source model.
    """
    # Load a table of (wavelength_in, flux_in) without any interpolation.
    table = config.load_table(
        config.source, ['wavelength', 'flux'], interpolate=False)
    # Get the position of this source.
    constants = config.get_constants(
        config.source.location, optional_names=['focal_x', 'focal_y'])
    if 'focal_x' in constants and 'focal_y' in constants:
        focal_xy_unit = constants['focal_x'].unit
        focal_xy = np.array([
            constants['focal_x'].value,
            constants['focal_y'].to(focal_xy_unit).value]) * focal_xy_unit
    else:
        focal_xy = None
    # Sky position is optional (and ignored) when x,y are specified.
    if hasattr(config.source.location, 'sky'):
        sky_position = config.get_sky(config.source.location)
    # Get the source profile on the sky.
    if hasattr(config.source, 'profile'):
        disk_fraction = config.source.profile.disk_fraction
        bulge_fraction = config.source.profile.bulge_fraction
        disk_shape = Profile(
            config.source.profile.disk_shape.half_light_radius,
            config.source.profile.disk_shape.minor_major_axis_ratio,
            config.source.profile.disk_shape.position_angle, sersic_index=1)
        bulge_shape = Profile(
            config.source.profile.bulge_shape.half_light_radius,
            config.source.profile.bulge_shape.minor_major_axis_ratio,
            config.source.profile.bulge_shape.position_angle, sersic_index=4)
    else:
        disk_fraction, bulge_fraction = 0, 0
        disk_shape, bulge_shape = None, None
    # Create a new Source object.
    source = Source(
        config.source.name, config.source.type, config.wavelength,
        table['wavelength'], table['flux'], disk_fraction,
        bulge_fraction, disk_shape, bulge_shape, focal_xy, sky_position,
        config.source.z_in, config.source.z_out, config.source.filter_name,
        config.source.ab_magnitude_out)
    if config.verbose:
        print("Initialized source '{0}' of type '{1}'."
              .format(source.name, source.type_name))
        if focal_xy is not None:
            print('Source located at (x, y) = ({0}, {1}).'
                  .format(*focal_xy))
        if sky_position is not None:
            radec = sky_position.transform_to('icrs')
            print('Source located at (ra, dec) = ({0}, {1}).'
                  .format(radec.ra, radec.dec))
        if config.source.z_out is not None:
            print('Redshift transformed from {0:.3f} to {1:.3f}.'
                  .format(config.source.z_in, config.source.z_out))
        if config.source.ab_magnitude_out is not None:
            print('Normalized to AB magnitude {0:.3f} in {1}.'
                  .format(config.source.ab_magnitude_out,
                          config.source.filter_name))
    return source
