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
