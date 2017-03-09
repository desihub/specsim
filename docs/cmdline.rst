Command-Line Program
====================

.. _quickspecsim:

QuickSpecSim
------------

This package includes a command-line program ``quickspecsim`` that simulates a
single spectrum and saves the results as a FITS file and/or plot. To see the
available command-line options use::

    quickspecsim --help

The ``--config`` parameter specifies the top-level :doc:`configuration file
</config>` to use and defaults to ``test``.  Without any arguments, the program
simulates a constant flux density source using the test atmosphere and
instrument models, producing the output::

    Median SNR in b camera = 1.165 / 0.5 Angstrom
    Median SNR in r camera = 0.941 / 0.5 Angstrom
    Median SNR in z camera = 0.742 / 0.5 Angstrom

Use the ``--output`` option to save the simulation results to a FITS file
with the following structure (as reported by `fitsinfo
<http://docs.astropy.org/en/stable/io/fits/usage/scripts.html
#module-astropy.io.fits.scripts.fitsinfo>`__)::

    No.    Name         Type      Cards   Dimensions   Format
    0    PRIMARY     PrimaryHDU       5   ()
    1    SIMULATED   BinTableHDU     45   63001R x 18C   [D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D]
    2    B           BinTableHDU     29   4760R x 10C   [D, D, D, D, D, D, D, D, D, D]
    3    R           BinTableHDU     29   4232R x 10C   [D, D, D, D, D, D, D, D, D, D]
    4    Z           BinTableHDU     29   4798R x 10C   [D, D, D, D, D, D, D, D, D, D]


Use the ``-save-plot`` option to visualize the simulation results,
for example::

    quickspecsim -c desi --save-plot sim.png

produces the following plot of a simulated 22nd AB magnitude reference source:

.. image:: _static/desi_ab22.png
    :alt: Simulated DESI response

A limited number of simulation parameters can be changed from the command line,
such as the exposure time, airmass and source magnitude.  For more substantial
changes to the simulation models, copy and edit an existing configuration file.

.. _quickfiberloss:

QuickFiberLoss
--------------

The ``quickfiberloss`` command-line program is primarily for performing
speed benchmarks of fiber acceptance fraction calculations using GalSim,
which are usually the rate-limiting step when these calculations are
performed on the fly.  To see the available command-line options use::

    quickfiberloss --help

The ``--config`` option has the same meaning as above, but only the
``instrument`` section of the configuration data will be used.  Normal usage
is, for example::

    quickfiberloss -n 100 --disk-fraction 0.5 --bulge-fraction 0.5
    Elapsed for 100 targets = 58.286 s, Rate = 582.861 ms/target

To measure the speed up for disk-only galaxies, try::

    quickfiberloss -n 100 --disk-fraction 1.0 --bulge-fraction 0.0
    Elapsed for 100 targets = 1.708 s, Rate = 17.084 ms/target
