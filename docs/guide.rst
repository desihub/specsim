.. |Ang| replace:: :math:`\text{\AA}`

User's Guide
============

A simulation has the following main components, as illustrated in the figure
below:

 - An astrophysical source.
 - A model of the sky and atmosphere.
 - An instrument, consisting of a telescope and one or more cameras.

An observation is specified by:

 - An exposure time :math:`\Delta t`.
 - An observing `airmass <https://en.wikipedia.org/wiki/Air_mass_(astronomy)>`__
   :math:`X`.

The table below lists all of the simulation modeling parameters described
in this guide, and specifies their units and which component they are associated
with.

.. figure:: img/overview.*
    :alt: Simulation overview diagram

    *Overview of the simulation components and parameters.  Camera components are
    shown in green, telescope components in gray, sky / atmospheric components
    in light blue, and source components in red.*

Source Model
------------

An astrophysical source is represented by its spectral energy distribution (SED)
:math:`s(\lambda)` and its profile.  We assume that a source's profile is
independent of its SED.

Atmosphere Model
----------------

The atmosphere adds its own emission spectrum :math:`b(\lambda)` to that of the
source and then both are attenuated by their passage through the atmosphere by
the extinction factor:

.. math::

    10^{-e(\lambda) X / 2.5}

where :math:`X` is the airmass of the observation. The atmosphere model also
specifies a point-spread function (PSF).

Instrument Model
----------------

The instrument consists of a telescope and one or more cameras.  The telescope
is described by the unobscured collecting area of its primary mirror :math:`A`,
used to normalize the source and sky fluxes, and the input face area :math:`a`
of its individual fibers, used to normalize the sky flux. The telescope also
has an optical PSF that is convolved with the atmospheric PSF and the source
profile to determine the profile of light incident upon the fiber entrance face
in the focal plane.  The overlap between this convolved profile and the fiber
aperture determines the fiber acceptance fraction :math:`f_S(\lambda)`. The
resulting flux :math:`F(\lambda)` in erg/|Ang| entering the fiber is:

.. math::

    F(\lambda) = \left[ s(\lambda) + a\, b(\lambda) \right] f_S(\lambda) A \Delta t

where :math:`\Delta t` is the exposure time.

All calculations are performed on a fine wavelength grid :math:`\lambda_j` that
covers the response of all cameras with a spacing that is smaller than the
wavelength extent of a single pixel by at least a factor of five.  The number of
photons entering the fiber with wavelengths :math:`\lambda_j \le \lambda <
\lambda_{j+1}` is then given by:

.. math::

    N^{\gamma}_j = \frac{\Delta \lambda_j}{h c \overline{\lambda}_j} F(\overline{\lambda}_j)

with wavelength bin widths and centers:

.. math::

    \Delta \lambda_j \equiv \lambda_{j+1} - \lambda_j \quad , \quad
    \overline{\lambda}_j \equiv (\lambda_j + \lambda_{j+1})/2 \; .

The remaining calculations are performed separately for each camera, indexed
by :math:`i`. The throughput function :math:`T_i(\lambda)` gives the combined
wavelength-dependent probability of transmission through the fiber and camera
optics, and then detection in the camera sensor resulting in a single
electron-hole pair.

The distribution of generated electron-hole pairs over sensor pixels is
determined by the camera's dispersion function :math:`d_i(\lambda)` and
effective wavelength resolution :math:`\sigma_i(\lambda)`.  We first apply
resolution effects using a convolution matrix :math:`R_{jk}`, resulting in:

.. math::

    N^{eh}_{ik} = \sum_j R_{jk} N^{\gamma}_j T_i(\lambda_j)

electron-hole pairs generated in camera :math:`j` and associated with the
*detected* wavelength :math:`\overline{\lambda}_k`.  Next, the continuous pixel
coordinate (in the wavelength direction) associated with the detected wavelength
is calculated as :math:`d(\overline{\lambda}_k)` and used to re-bin
electron-hole counts from the fine detected wavelength grid :math:`N^{eh}_{ik}`
to sensor pixels :math:`N^{eh}_{ip}`.

Finally, the sensor electronics response is characterized by a gain :math:`G_i`,
dark current :math:`I_{dk}` and readout noise :math:`\sigma_{ro,i}`.  The
resulting signal in detected electrons is given by:

.. math::

    N^{det}_{ip} = G N^{eh}_{ip} + I_{dk,i} n_{ip} \Delta t

with a corresponding variance:

.. math::

    V^{det}_{ip} = N^{det}_{ip} + \sigma^2_{ro,i} n_{ip}

where :math:`n_{ip}` is the effective trace width in pixels for wavelength pixel
:math:`p`, and we assume that read noise is uncorrelated between pixels (so its
variance scales with :math:`n_{ip}`).

.. |none| replace:: :math:`\text{---}`
.. |sunit| replace:: :math:`\text{erg}/(\text{s cm}^2 \AA)`
.. |bunit| replace:: :math:`\text{erg}/(\text{s cm}^2 \AA)/\text{arcsec}`
.. |eunit| replace:: :math:`\text{mags}/\text{airmass}`
.. |Gunit| replace:: :math:`\text{det.elec.} / \text{e-h pair}`
.. |Iunit| replace:: :math:`\text{det.elec.} / \text{s} / \text{pixel}`
.. |runit| replace:: :math:`\text{det.elec.} / \text{pixel}`

+----------------------+-----------------------+---------------+-------------------------+
| Parameter            | Units                 | Component     | Description             |
+======================+=======================+===============+=========================+
| :math:`\Delta t`     | :math:`\text{s}`      | Observation   | Exposure time           |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`X`            | |none|                | Observation   | Observing airmass       |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`\lambda_i`    | |Ang|                 | Simulation    | Fine wavelength grid    |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`s(\lambda)`   | |sunit|               | Source        | Source SED              |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`b(\lambda)`   | |bunit|               | Atmosphere    | Sky surface brightness  |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`e(\lambda)`   | |eunit|               | Atmosphere    | Atmospheric extinction  |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`A`            | :math:`\text{cm}^2`   | Telescope     | Primary unobscured area |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`a`            | :math:`\text{arcsec}` | Telescope     | Fiber entrance area     |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`f_S(\lambda)` | |none|                | Src+Atmos+Tel | Fiberloss fraction      |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`T_i(\lambda)` | |none|                | Tel+Camera    | Transmission throughput |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`I_{dk,i}`     | |Iunit|               | Camera        | Sensor darkcurrent      |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`G_i`          | |Gunit|               | Camera        | Readout gain            |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`\sigma_{ro,i}`| |runit|               | Camera        | Readout noise           |
+----------------------+-----------------------+---------------+-------------------------+
| :math:`n_{ip}`       | :math:`\text{pixels}` | Camera        | Trace width             |
+----------------------+-----------------------+---------------+-------------------------+
