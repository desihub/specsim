Contributing to the SpecSim Package
===================================

This package is maintained and developed under the github [desihub organization](https://github.com/desihub) but, unlike most packages there, is designed to be a general purpose tool that can be configured to simulate any fiber spectrograph.  As such, it uses the [astropy affiliated package template](http://docs.astropy.org/en/latest/development/affiliated-packages.html) rather than the [DESI software template](https://github.com/desihub/desitemplate).

Release Checklist
-----------------

Follow [these instructions](http://docs.astropy.org/en/latest/development/affiliated-packages.html#releasing-an-affiliated-package).  Tagged releases are distributed at https://pypi.python.org/pypi/specsim

Updating the Astropy Affiliated Template
----------------------------------------

This package was created using the ["Managing the template files via git" option]
("http://docs.astropy.org/en/latest/development/affiliated-packages.html#managing-the-template-files-via-git").  Instructions for updating template files are provided [here](http://docs.astropy.org/en/latest/development/affiliated-packages.html#id3), but don't use them since they generate a lot of merge conflicts that take a while to sort out.  Instead, use the following steps:

* `git fetch --no-tags template` (we do not want to export tags of the template pacakge).
* `git diff template/master`: review what files have changed and how.
* For each new template file that you want to add:
  * `git checkout template/master file1`
  * `git add file1`
* For each changed template file that you want to update:
  * `git checkout template/master file2` `git diff HEAD^ file2`
  * Edit file by hand, referring to the diff to select which changes to revert.
  * `git add file2`

Lines in `.travis.yml` that you should not change unless you are certain:
```
- PIP_DEPENDENCIES='speclite'
- CONDA_DEPENDENCIES='scipy pyyaml matplotlib'
- CONDA_CHANNELS='astropy-ci-extras'
- SETUP_XVFB=True
- if [[ $SETUP_CMD == *coverage* ]]; then coveralls --rcfile='specsim/tests/coveragerc'; fi
```

Files that you should normally never update from the template: `CHANGES.rst`, `README.rst`, `TEMPLATE_CHANGES.md`.

To update to a more recent tag of `astropy_helpers`, use something like:
```
cd astropy_helpers
git fetch
git checkout v1.3.1
cd ..
git add astropy_helpers
git commit -m 'Update to astropy_helpers v1.3.1'
```
