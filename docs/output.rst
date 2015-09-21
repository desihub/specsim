Output Format
=============

The results of a simulation are output as a `numpy structured array
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ tabulated on the
nominal co-add wavelength grid, with the columns listed below. Columns indexed
by `[j]` are repeated for each simulated camera.

.. |Ang| replace:: :math:`\text{\AA}`
.. |none| replace:: :math:`\text{---}`
.. |funit| replace:: :math:`10^{-17} \text{erg}/(\text{s cm}^2 \AA)`
.. |iunit| replace:: :math:`(10^{-17} \text{erg}/(\text{s cm}^2 \AA))^2`
.. |eunit| replace:: :math:`\text{elec}`

+--------------+---------+-------------------------------------------------+
| Column Name  | Units   | Description                                     |
+==============+=========+=================================================+
| `wave`       | |Ang|   | Co-add nominal wavelength grid                  |
+--------------+---------+-------------------------------------------------+
| `srcflux`    | |funit| | Source flux                                     |
+--------------+---------+-------------------------------------------------+
| `obsflux`    | |funit| | Observed co-add flux                            |
+--------------+---------+-------------------------------------------------+
| `ivar`       | |iunit| | Inverse variance of `obsflux`                   |
+--------------+---------+-------------------------------------------------+
| `snrtot`     | |none|  | Total SNR of co-added spectrum                  |
+--------------+---------+-------------------------------------------------+
| `nobj[j]`    | |none|  | Mean number of observed source photons          |
+--------------+---------+-------------------------------------------------+
| `nsky[j]`    | |none|  | Mean number of observed sky photons             |
+--------------+---------+-------------------------------------------------+
| `rdnoise[j]` | |eunit| | RMS read noise in camera `j`                    |
+--------------+---------+-------------------------------------------------+
| `dknoise[j]` | |eunit| | RMS dark current shot noise in camera `j`       |
+--------------+---------+-------------------------------------------------+
| `snr[j]`     | |none|  | SNR in camera `j`                               |
+--------------+---------+-------------------------------------------------+

The nominal co-add wavelength grid is obtained by downsampling the fine
wavelength grid used internally by an integer factor.  The fine and downsampled
grids are both configured by :ref:`command-line options <config_command_line>`.
