Configuration
=============

Configuration options specify the instrument and atmosphere to simulate.

Instrument
----------

An instrument consists of an optical system illuminating an array of fibers
positioned at the focal plane.  Fibers illuminate a set of cameras.

The instrument is configured with a list of design parameters.

The focal plane is simulated as a set of fiber loss models for different
source types.

Each camera is specified by its wavelength-dependent point-spread function
(PSF) and throughput.

Atmosphere
----------

The atmosphere is configured by its emission spectrum and the extinction due to
its absorption.
