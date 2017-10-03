Installation
============

The specsim package is compatible with python versions 2.6, 2.7, 3.3 and 3.4.

To see what version, if any, you already have installed use::

    pip show specsim

Install Latest Stable Release
-----------------------------

Install the latest stable release using::

    pip install specsim

On systems where you do not have the privileges required to install python
packages, use instead::

    pip install specsim --user

The documentation of the latest stable release is `here
<http://specsim.readthedocs.io/en/stable/>`__. The required dependencies listed
below will be automatically installed by this command.

To update to a newer stable release after your initial install, use::

    pip install specsim --upgrade

To uninstall any previously installed stable release use::

    pip uninstall specsim

This method will also uninstall a developer version.  You do not need to
uninstall before upgrading to the latest stable release.

Install Latest Developer Version
--------------------------------

Alternatively, you can install the latest developer version from github::

    github clone https://github.com/desihub/specsim.git
    cd specsim
    python setup.py install

On systems where you do not have the privileges required to install python
packages, use instead::

    python setup.py install --user

The documentation of the latest developer release is `here
<http://specsim.readthedocs.io/en/latest/>`_. The required dependencies listed
below will be automatically installed by the `setup.py` step above.

Any changes you make to your git cloned package after running the `setup.py`
step will not affect the installed version.  If you want your changes to
apply directly to the installed version, use a "live install" instead::

    python setup.py develop

On systems where you do not have installation privileges, use::

    python setup.py develop --user

To stop using your git clone as a live install, use::

    python setup.py develop --uninstall

Required Dependencies
---------------------

The recommended way to obtain and maintain all of these dependencies is to use
a scientific python distribution such as  `anaconda
<https://store.continuum.io/cshop/anaconda/>`__

* `numpy <http://www.numpy.org/>`__ (version >= 1.8)
* `scipy <http://www.scipy.org/scipylib/index.html>`__
* `astropy <http://www.astropy.org/>`__
* `pyyaml <http://pyyaml.org/wiki/PyYAML>`__
* `speclite <http://speclite.readthedocs.io>`__ (version >= 0.8)

Optional Dependency: matplotlib
-------------------------------

The `matplotlib <http://matplotlib.org>`__ package enables optional plotting
capabilities and should already be installed if you are using a scientific
python distribution.

Optional Dependency: galsim
---------------------------

The `galsim <https://github.com/GalSim-developers/GalSim/wiki>`__ package
enables the most flexible calculations of
:doc:`fiber acceptance fractions <fiberloss>`.
