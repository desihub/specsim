API Reference
=============

You will normally only need to import two modules from this package::

    import specsim.config
    import specsim.simulator

The simplest possible simulation involves loading a configuration, initializing
a simulator, and simulating one exposure::

    config = specsim.config.load_config('test')
    simulator = specsim.simulator.Simulator(config)
    results = simulator.simulate()

In this example, the entire simulation is configured by the loaded
configuration file.  To use a different configuration, either copy and edit
this file or else change parameters programmatically before initializing
the simulator, for example::

    config = specsim.config.load_config('test')
    config.atmosphere.airmass = 1.5
    config.source.filter_name = 'sdss2010-r'
    config.source.ab_magnitude_out = 22.5
    simulator = specsim.simulator.Simulator(config)
    results = simulator.simulate()

Some parameters can also be changed via an initialized simulator, without
repeating the initialization step, for example::

    config = specsim.config.load_config('test')
    simulator = specsim.simulator.Simulator(config)
    results1 = simulator.simulate()
    simulator.atmosphere.airmass = 1.5
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

.. _source-api:
.. automodapi:: specsim.source
    :no-inheritance-diagram:

.. _simulator-api:
.. automodapi:: specsim.simulator
    :no-inheritance-diagram:

.. _transform-api:
.. automodapi:: specsim.transform
    :no-inheritance-diagram:

.. _driver-api:
.. automodapi:: specsim.driver
    :no-inheritance-diagram:
