Command-Line Program
====================

This page describes the command-line program for running simuations...

Command-Line Options
--------------------

The following table summarizes the command-line options used to configure
the simulation that will be performed.  Use the `--help` option for more
details.

+------------------+---------+------------+---------------------------------------------------------------+
| Option           | Default | Component  | Description                                                   |
+==================+=========+============+===============================================================+
| exptime          | (YAML)  | Observer   | Exposure time in seconds (use YAML `exptime` as default)      |
+------------------+---------+------------+---------------------------------------------------------------+
| airmass          | 1.0     | Observer   | Observing airmass                                             |
+------------------+---------+------------+---------------------------------------------------------------+
| model            |         | Source     | Source profile model use for fiber acceptance fraction        |
+------------------+---------+------------+---------------------------------------------------------------+
| infile           |         | Source     | Name of file containing wavelength (Ang) and                  |
|                  |         |            | flux (1e-17 erg/cm^2/s/Ang) columns                           |
+------------------+---------+------------+---------------------------------------------------------------+
| infile-wavecol   | 0       | Source     | Index of infile column containing wavelengths                 |
|                  |         |            | (starting from 0)                                             |
+------------------+---------+------------+---------------------------------------------------------------+
| infile-fluxcol   | 1       | Source     | Index of infile column containing fluxes                      |
+------------------+---------+------------+---------------------------------------------------------------+
| truncated        | False   | Source     | Assume zero flux outside of source spectrum wavelength range  |
+------------------+---------+------------+---------------------------------------------------------------+
| ab-magnitude     |         | Source     | Source spectrum flux rescaling, e.g. g=22.0 or r=21.5         |
+------------------+---------+------------+---------------------------------------------------------------+
| redshift-to      |         | Source     | Redshift source spectrum to this value                        |
+------------------+---------+------------+---------------------------------------------------------------+
| redshift-from    | 0.0     | Source     | Redshift source spectrum from this value                      |
+------------------+---------+------------+---------------------------------------------------------------+
| sky              | 'dark'  | Atmosphere | Sky conditions to simulate (dark | gray | bright)             |
+------------------+---------+------------+---------------------------------------------------------------+
| nread            | 1.0     | Simulation | Scale readout noise variance by this factor                   |
+------------------+---------+------------+---------------------------------------------------------------+
| min-wavelength   | 3500.3  | Simulation | Minimum wavelength to simulate in Angstroms                   |
+------------------+---------+------------+---------------------------------------------------------------+
| max-wavelength   | 9999.7  | Simulation | Maximum wavelength to simulate in Angstroms                   |
+------------------+---------+------------+---------------------------------------------------------------+
| wavelength-step  | 0.1     | Simulation | Linear spacing of simulation wavelength grid in Angstroms     |
+------------------+---------+------------+---------------------------------------------------------------+
