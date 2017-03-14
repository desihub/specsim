API Reference
=============

The simplest possible simulation involves initializing a simulator and
simulating one exposure, for example::

    import specsim.simulator

    simulator = specsim.simulator.Simulator('desi')
    results = simulator.simulate()

In this example, the entire simulation is configured by the contents of the file
``specsim/data/config/desi.yaml``. To use a different configuration, either
copy and edit this file or else change parameters programmatically before
initializing the simulator, for example::

    import specsim.config
    import specsim.simulator

    config = specsim.config.load_config('desi')
    config.atmosphere.airmass = 1.5
    config.source.filter_name = 'sdss2010-r'
    config.source.ab_magnitude_out = 22.5

    simulator = specsim.simulator.Simulator(config)
    results = simulator.simulate()

Many parameters can also be changed via an initialized simulator, without
repeating the initialization step, for example::

    import specsim.simulator

    simulator = specsim.simulator.Simulator('desi')
    results1 = simulator.simulate()

    simulator.atmosphere.airmass = 1.5
    simulator.observation.exposure_time = 20 * u.min
    simulator.source.update_out(filter_name='sdss2010-r', ab_magnitude_out=21.0)
    results2 = simulator.simulate()

.. _config-api:
.. automodapi:: specsim.config
    :no-inheritance-diagram:

.. _atmosphere-api:
.. automodapi:: specsim.atmosphere
    :no-inheritance-diagram:

.. _instrument-api:
.. automodapi:: specsim.instrument
    :no-inheritance-diagram:

.. _camera-api:
.. automodapi:: specsim.camera
    :no-inheritance-diagram:

.. _source-api:
.. automodapi:: specsim.source
    :no-inheritance-diagram:

.. _observation-api:
.. automodapi:: specsim.observation
    :no-inheritance-diagram:

.. fiberloss-api:
.. automodapi:: specsim.fiberloss
    :no-inheritance-diagram:

.. _simulator-api:
.. automodapi:: specsim.simulator
    :no-inheritance-diagram:

.. _transform-api:
.. automodapi:: specsim.transform
    :no-inheritance-diagram:
