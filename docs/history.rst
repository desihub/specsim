History
=======

The initial version of this code was based on IDL code "Simple S/N calculator
for DESI spectra" written for DESI and available `here
<https://desi.lbl.gov/svn/code/desimodel/tags/0.4.2/pro/desi_quicksim.pro>`__.
Early versions of this package were maintained in the
same SVN package and aimed to produce identical results.

This package originated as a DESI-specific simulation tool but, as of v0.3,
has no code dependencies on DESI software and makes no hardcoded assumptions
that are specific to DESI.  Instead, any fiber spectrograph and observing
conditions can be configured.

Backwards Compatibility
-----------------------

The following recipes all give identical results in their output file ``ab22.dat``
and assume that the DESIMODEL environment variable has been set.

First, using the final SVN python version (revision 2134)::

    cd $DESIMODEL
    python bin/quicksim.py --infile data/spectra/spec-ABmag22.0.dat --model qso \
        --verbose --show-plot --outfile ab22.dat

Next, using the first release (v0.1) after moving to a dedicated package on github::

    export SPECSIM_MODEL=$DESIMODEL
    ln -s $SPECSIM_MODEL/data/throughput specsim/data/throughput
    ln -s $SPECSIM_MODEL/data/spectra specsim/data/spectra
    quickspecsim --infile specsim/data/spectra/spec-ABmag22.0.dat --model qso \
        --verbose --show-plot --outfile ab22.dat

Finally, using the first release (v0.3) that is completely independent of DESI software::

    quickspecsim -c desi -o ab22.dat --show-plot

Later versions include improvements that give better but no longer identical results.
