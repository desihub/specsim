Command-Line Program
====================

This package includes a command-line program `quickspecsim` that simulates a
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
