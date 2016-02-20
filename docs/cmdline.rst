Command-Line Program
====================

This package includes a command-line program `quickspecsim` that simulates a
single spectrum and saves the results as a text file and/or plot. To see the
available command-line options use::

    quickspecsim --help

The ``--config`` parameter specifies the top-level :doc:`configuration file
</config>` to use and defaults to ``test``.  Without any arguments, the program
simulates a constant flux density source using the test atmosphere and
instrument models, producing the output::

    Median S/N = 2.107, Total (S/N)^2 = 18068.7

Use the ``--show-plot`` and ``-save-plot`` options to visualize the simulation
results, for example::

    quickspecsim -c desi --show-plot

produces the following plot of a simulated 22nd AB magnitude reference source:

.. image:: _static/desi_ab22.png
    :alt: Simulated DESI response

A limited number of simulation parameters can be changed from the command line,
such as the exposure time, airmass and source magnitude.  For more substantial
changes to the simulation models, copy and edit an existing configuration file.
