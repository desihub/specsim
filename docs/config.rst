Configuration
=============

This section describes how a simulation is configured using information read
from files and command-line options.  For an overview of how a simulation is
performed and what parameters it requires see the :doc:`/guide`. Configuration
options specify the instrument and atmosphere to simulate, in addition to the
source properties and observing parameters.

.. _config_yaml:

YAML
----

The table below lists the parameters that are required in a `YAML file
<http://https://en.wikipedia.org/wiki/YAML>`__ describing the instrument.
Note that the `ccd.*` prefixes are repeated once per camera.

+--------------------------+------------------------------------------------------+
| YAML Name                | Description                                          |
+==========================+======================================================+
| `area.geometric_area`    | Unobscured effective area of primary mirror in m^2   |
+--------------------------+------------------------------------------------------+
| `fibers.diameter_arcsec` | Average fiber diameter in arcsecs                    |
+--------------------------+------------------------------------------------------+
| `ccd.*.readnoise`        | RMS readout noise in electrons per pixel             |
+--------------------------+------------------------------------------------------+
| `ccd.*.darkcurrent`      | Average dark current in electrons/hour/pixel         |
+--------------------------+------------------------------------------------------+
| `ccd.*.gain`             | Readout gain in electrons per ADU                    |
+--------------------------+------------------------------------------------------+
| `exptime`                | Nominal exposure time in seconds                     |
+--------------------------+------------------------------------------------------+

.. _config_data:

Tabulated Data
--------------

The table below lists the FITS binary table files required to configure a
simulation.  The `*` pattern specifies a camera name (B, R, Z).

+-------------------------------------+--------------+---------------------------------------+
| Filename                            |  HDU Name    | Column Names                          |
+=====================================+==============+=======================================+
| `data/specpsf/psf-quicksim.fits`    | `QUICKSIM-*` | wavelength, angstroms_per_row,        |
|                                     |              | fwhm_wave, fwhm_spatial, neff_spatial |
+-------------------------------------+--------------+---------------------------------------+
| `data/throughput/thru-*.fits`       | `THROUGHPUT` | wavelength, throughput                |
+-------------------------------------+--------------+---------------------------------------+

The following ASCII data files are also required, where `<S>` denotes a source
model (elg, lrg, qso, ...) and `<C>` denotes optional sky conditions (-grey, -bright).

+------------------------------------------+---------------------------------+
| Filename                                 | Column Names                    |
+==========================================+=================================+
| `data/throughput/fiberloss-<S>.dat`      | Wavelength, FiberAcceptance     |
+------------------------------------------+---------------------------------+
| `data/spectra/spec-sky<C>.dat`           | WAVE, FLUX                      |
+------------------------------------------+---------------------------------+
| `data/spectra/ZenithExtinction-KPNO.dat` | WAVELENGTH, EXTINCTION          |
+------------------------------------------+---------------------------------+

.. _config_command_line:

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
