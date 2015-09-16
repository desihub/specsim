specsim
=======

Quick simulations of fiber spectrograph response.

The following recipe gives identical results to the original desimodel quicksim::

    # Using desimodel (svn revision 2134)
    export DESIMODEL=`pwd`
    python bin/quicksim.py --infile data/spectra/spec-ABmag22.0.dat --model qso --verbose --show-plot --outfile ab22.dat

    # Using specsim (git tag 0.1)
    export SPECSIM_MODEL=$DESIMODEL
    ln -s $DESIMODEL/data/throughput specsim/data/throughput
    ln -s $DESIMODEL/data/spectra specsim/data/spectra
    quickspecsim --infile specsim/data/spectra/spec-ABmag22.0.dat --model qso --verbose --show-plot --outfile ab22.dat


Status reports for developers
-----------------------------

.. image:: https://travis-ci.org/desihub/specsim.png?branch=master
    :target: https://travis-ci.org/desihub/specsim
    :alt: Test Status

.. image:: https://readthedocs.org/projects/specsim/badge/?version=latest
    :target: https://readthedocs.org/projects/specsim/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/desihub/specsim/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/desihub/specsim?branch=master
    :alt: Coverage Status

.. image:: https://img.shields.io/pypi/v/specsim.svg
    :target: https://pypi.python.org/pypi/specsim
    :alt: Distribution Status
