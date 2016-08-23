0.5 (unreleased)
----------------

- Fix github issues #41, #42.
- Update to latest astropy affiliated package template.
- Drop support for python 2.6 and add support for python 3.5.
- Add testing against LTS release of astropy.
- Drop testing against numpy 1.6 and add numpy 1.11.
- Update readthedocs URLs (.org -> .io).

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
