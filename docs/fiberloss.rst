Fiber Acceptance Fraction Calculations
======================================

Overview
--------

Fiber acceptance refers to the fraction of photons incident on the focal plane
from an astrophysical source that enter the fiber.  The fiber acceptance
fraction is also referred to as the "fiberloss", and depends on:

* The source profile on the sky, modeled as a sum of point-like, disk
  (Sersic n=1), and bulge (Sersic n=4) components.
* The atmospheric PSF, modeled with a Moffat function with an adjustable beta
  value, and assuming a Kolmogorov exponent (-1/5) for the wavelength dependence
  of the astmospheric seeing FWHM.
* The instrumental PSF which has both blur and offset components and generally
  depends on position in the focal plane and wavelength.
* The mapping of sky coordinates to physical coordinates at the fiber entrance
  face, which are expressed via radial and azimuthal plate scales, and generally
  transform a round fiber into an elliptical view of the sky.

Further details specific to DESI are provided in `DESI-doc-2720 <https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2720>`_.  The
jupyter notebook in ``docs/nb/FiberlossExamples.ipynb`` was used to create the
plots for this document and provide code examples for fiberloss calculations.

Configuration
-------------

The fiberloss calculation combines many different aspects of the simulation,
so its configuration data is spread among different components: ``atmosphere``,
``instrument``, ``source`` and ``observation``.

Details of how fiberloss should be calculated are specified in the
``instrument.fiberloss`` configuration.  There are currently two supported
methods selected by the value of ``instrument.fiberloss.method``:

* The ``table`` method reads fiberloss arrays for different object types
  (QSO, LRG, etc) from files during initialization. The ``source.type``
  parameter is then used to select the appropriate array.  This mode is
  fast but inflexible.  Configuration parameters related to the source profile,
  atmospheric seeing, the instrumental PSF, and plate scales are not used
  to calculate the fiberloss with this method.
* The ``galsim`` method calculates fiberloss arrays on the fly using the
  `galsim <https://github.com/GalSim-developers/GalSim/wiki>`_ external package
  to build and convolve models of the source and
  the atmospheric and instrumental PSFs.  Models are built in physical sky
  coordinates, using the plate scales to transform the on-sky source and
  atmospheric models.  This mode is the most flexible and slowest, and uses
  all of the relevant configuration parameters.
* The ``fastsim`` method is based on an interpolation of fiber acceptance values
  pre-computed with ``galsim`` using the script ``fitgalsim``.
  The interpolation parameters/axis are source profile (point source, exponential disk, De Vaucouleurs bulge), sigma (atmosphere+telescope blur, in um on focal surface), fiber offset from source (in um on focal surface), and half light radius (in arcsec) from extended source. This interpolation is performed with the class ``fastfiberacceptance.FastFiberAcceptance``. The accuracy is about 1 percent, which is sufficient for simulations.
  
Refer to the comments in the sample configuration files for more information
about the available parameters.

GalSim Dependency
-----------------

The GalSim package is only required when the ``galsim`` method is selected.
Fiberloss arrays calculated with GalSim can be saved for later table-based
simulations.  For example, first run the :ref:`quickspecsim` script to
use GalSim to calculate and save the fiberloss for a specific configuration
of the source, atmosphere, etc, with::

    quickspecsim -c desi-elg --save-fiberloss desi-elg-fiberloss

Next, update your configuration file to select the ``table`` method and read
the new tabulated fiberloss array::

    instrument:
    ...
    fiberloss:
        method: table
        format: ascii.ecsv
        paths:
            elg: .../desi-elg-fiberloss.ecsv

API
---

Refer to the :mod:`specsim.fiberloss` documentation for details on the
API for performing fiberloss calculations.  The source code of the
:ref:`quickfiberloss` script provides a simple example of calculating
fiberloss arrays with GalSim, starting from arrays of galaxy sizes and shapes.

GalSim Accuracy and Speed
-------------------------

The accuracy and speed of the GalSim calculations are primarily determined by
the ``instrument.fiberloss.num_wlen`` and ``instrument.fiberloss.num_pixels``
configuration parameters.  The first parameter determines the number of
wavelengths at which the calculations are performed, and the time scales
linearly with its value.  Lower values are faster but require more interpolation
in wavelength, leading to larger errors.  The second parameter determines the
resolution of the pixel grid used to cover a fiber in the calculations.  The
timing is non-linear in this parameter, with the default value of 16 giving
a local optimum speed, with relative errors in fiberloss at the 1e-3 level.

The :ref:`quickfiberloss` command-line script is primarily intended for
benchmarking different numerical parameters.
