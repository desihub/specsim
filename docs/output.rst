Simulation Outputs
==================

The results of a simulation are stored in several :class:`astropy tables
<astropy.table.Table>` that are overwritten after each call to :meth:`simulate
<specsim.simulator.Simulator.simulate>`.  The :meth:`plot
<specsim.simulator.Simulator.plot>` method provides a convenient visual
summary of all of these quantities.  See :doc:`/cmdline` for a sample plot.

The first output table is accessible as :attr:`simulated
<specsim.simulator.Simulator.simulated>` and tabulates fluxes and numbers of
photons and electrons on the high-resolution wavelength grid used internally
by the simulation.  Each camera's response is recorded in four columns with
names ending with the camera name (indicated by ``*`` below).

There is an additional table per camera that tabulates quantities on each
camera's output pixels.  These per-camera tables are stored in the
:attr:`camera_output <specsim.simulator.Simulator.camera_output>` list in
order of increasing wavelength.
Output pixel values are calculated from the high-resolution :attr:`simulated
<specsim.simulator.Simulator.simulated>` values by downsampling.

Wavelength and flux columns are stored with units, while numbers of photons
and electrons are dimensionless and represent mean expected values.

Simulated Output Table
----------------------

The table below defines the colums of the :attr:`simulated
<specsim.simulator.Simulator.simulated>` table.  The initial ``wavelength``
column has shape (nwlen) and the remaining columns all have shape
(nwlen, nfibers).

+----------------------------+------------------------------------------------+
| Column Name                | Description                                    |
+============================+================================================+
| ``wavelength``             | Central wavelength of each simulation bin      |
+----------------------------+------------------------------------------------+
| ``source_flux``            | Source flux above the atmosphere               |
+----------------------------+------------------------------------------------+
| ``fiberloss``              | Fraction of source flux entering the fiber     |
+----------------------------+------------------------------------------------+
| ``source_fiber_flux``      | Source flux into the fiber                     |
+----------------------------+------------------------------------------------+
| ``sky_fiber_flux``         | Sky flux into the fiber                        |
+----------------------------+------------------------------------------------+
| ``num_source_photons``     | Number of source photons entering the fiber    |
+----------------------------+------------------------------------------------+
| ``num_sky_photons``        | Number of sky photons entering the fiber       |
+----------------------------+------------------------------------------------+
| ``num_source_electrons_*`` | Number of source electrons recorded by the CCD |
+----------------------------+------------------------------------------------+
| ``num_sky_electrons_*``    | Number of sky electrons recorded by the CCD    |
+----------------------------+------------------------------------------------+
| ``num_dark_electrons_*``   | Number of dark current electrons in the CCD    |
+----------------------------+------------------------------------------------+
| ``read_noise_electrons_*`` | RMS read noise in electrons for the CCD        |
+----------------------------+------------------------------------------------+

Note that the ``num_source_electrons_*`` and ``num_sky_electrons_*`` arrays are
normally convolved with the appropriate camera resolution, but this convolution
is not performed by a :class:`specsim.simulator.Simulator` initialized using the
option ``camera_output = False``.  This can be useful when specsim is
being used to calculate inputs to a more detailed pixel-level simulation.

Camera Output Tables
--------------------

The table below defines the columns of each table listed in
:attr:`camera_output <specsim.simulator.Simulator.camera_output>`.
The initial ``wavelength``
column has shape (nwlen_out) and the remaining columns all have shape
(nwlen_out, nfibers).

These tables can require a lot of memory so, if they are not needed,
initialize your :class:`specsim.simulator.Simulator` using the option
``camera_output = False`` to save space and speed up simulations.

+----------------------------+------------------------------------------------+
| Column Name                | Description                                    |
+============================+================================================+
| ``wavelength``             | Central wavelength of each output pixel        |
+----------------------------+------------------------------------------------+
| ``num_source_electrons``   | Number of source electrons recorded by the CCD |
+----------------------------+------------------------------------------------+
| ``num_sky_electrons``      | Number of sky electrons recorded by the CCD    |
+----------------------------+------------------------------------------------+
| ``num_dark_electrons``     | Number of dark current electrons in the CCD    |
+----------------------------+------------------------------------------------+
| ``read_noise_electrons``   | RMS read noise in electrons for the CCD        |
+----------------------------+------------------------------------------------+
| ``random_noise_electrons`` | Random CCD noise realization in electrons      |
+----------------------------+------------------------------------------------+
| ``variance_electrons``     | Variance of the total number of electrons      |
+----------------------------+------------------------------------------------+
| ``flux_calibration``       | Calibration from CCD electrons to source flux  |
+----------------------------+------------------------------------------------+
| ``observed_flux``          | Perfectly calibrated observed flux             |
+----------------------------+------------------------------------------------+
| ``flux_inverse_variance``  | Inverse variance of ``observed_flux``          |
+----------------------------+------------------------------------------------+

The ``observed_flux`` and ``flux_inverse_variance`` columns are calculated
assuming perfect flux calibration as::

    observed_flux = flux_calibration * num_source_electrons
    flux_inverse_variance = flux_calibration ** -2 * variance_electrons ** -1

The ``random_noise_electrons`` column is zeroed during each call to
:meth:`simulate <specsim.simulator.Simulator.simulate>`, and can then be
optionally filled (repeatedly when useful) with :meth:`generate_random_noise
<specsim.simulator.Simulator.generate_random_noise>`.  To propagate a noise
realization in electrons to flux, use::

    random_noise_flux = flux_calibration * random_noise_electrons

To calculate the signal-to-noise ratio (SNR) in each camera output pixel use::

    SNR = num_source_electrons / sqrt(variance_electrons)
