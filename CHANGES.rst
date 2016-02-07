0.3 (unreleased)
----------------

- Read files using astropy.table.Table.read() instead of numpy.loadtxt()
  and astropy.io.fits.read().

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
