0.11 (2017-10-10)
-----------------

- Implement fast fiberloss calculator that interpolates GalSim results (#78).
- Add uncorrelated random offsets (#81)

0.10 (2017-10-03)
-----------------

- Update to astropy_helpers v2.0.1 (#79).
- Handle non-backwards compatible changes to astropy.constants in astropy 2.0.
- Handle deprecated longitude, latitude attributes for astropy >= 2.0.
- Remove tests against python 3.3 and numpy 1.7.

0.9 (2017-05-11)
----------------

- Fix Simulator output table units with astropy 1.3.2 (#62).
- Add support for simulating calibration exposures (#54).
- Add wavelength min/max options to Simulator.plot().
- Better support for alternate simulation wavelength grids (PR #60).
- Add Simulator option to not create and fill per-camera output tables (PR #67).

0.8 (2017-03-25)
----------------

This version adds the ability to specify per-fiber positions and source
properties.  Fiber acceptance fractions can now be calculated on the fly
(using GalSim) to simulate realistic throughput variations across the
focal plane.

- Add support and travis testing for python 3.6.
- Add fiberloss module with support for more realistic fiberloss calculations.
- Add optional dependency on galsim (only required for on-the-fly fiberloss
  calculations).
- Add jupyter notebooks to document the Simulator and fiberloss calculations.
- Update the simulate() method to take optional array parameters specifying
  per-fiber positions and source properties.
- Allow the observation time and pointing to be changed after initialization.

0.7 (2016-12-02)
----------------

- Add scattered moon surface brightness attribute to Moon model.
- This is a minor release with a change required for DESI survey simulations.

0.6 (2016-11-09)
----------------

- Fix github issues #43, #49.
- Create DOI for v0.5 https://doi.org/10.5281/zenodo.154130.
- Re-implement Camera.get_output_resolution_matrix to return a sparse
  matrix, which uses significantly less memory and runs 4-5x faster.

0.5 (2016-09-12)
----------------

- Fix github issues #41, #42, #47.
- Update to latest astropy affiliated package template.
- Drop support for python 2.6 and add support for python 3.5 (w/o 2to3).
- Add testing against LTS release of astropy.
- Drop testing against numpy 1.6 and add numpy 1.11.
- Update readthedocs URLs (.org -> .io).
- Implement optional radial and azimuthal plate scales varying with radius.
- Add observation config parameters to support sky <-> xy transforms.
- Refactor instrument module into separate instrument and camera modules.

0.4 (2016-03-08)
----------------

- Fix github issues #1, #4, #8, #9, #17, #18, #26.
- Implement workaround for missing IERS data in sidereal time calculation.
- Silence astropy warnings for non-standard units in FITS files.
- Clean up simulator module to streamline its use by desisim.
- Refactor instrument model to handle downsampling to output pixels.
- Implement scattered moon component of sky brightness (#9).
- Apply extinction to sky emission by default (#8).

0.3 (2016-02-19)
----------------

This version introduces some significant API changes in order to make the
code instrument agnostic and facilitate future algorithm improvements.
There are no changes yet to the underlying algorithms and assumptions, so
results using the new desi.yaml config should be identical to v0.2.

- Add new config module for flexible specification of all simulation options,
  including the instrument model definition.
- Create config files for DESI and unit testing.
- Refactor to make code instrument-agnostic, with no dependencies on
  DESI packages.
- Read files using astropy.table.Table.read() instead of numpy.loadtxt()
  and astropy.io.fits.read().
- Remove unused sources, spectrum modules.
- Rename quick.Quick to simulator.Simulator.
- Add speclite dependency.

0.2 (2015-12-18)
----------------

- Add the transform module for coordinate transformations between the sky,
  alt-az, and the focal plane.
- Minor improvements to sparse resolution matrix edge effects.
- Provide per-camera flux and ivar predictions.

0.1 (2015-09-16)
----------------

- Initial release after migration from desimodel SVN.
- Gives identical results to quicksim.
