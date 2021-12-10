`kpdvdb`: KayPENTAX Disordered Voice Database Reader
====================================================

|pypi| |status| |pyver| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/kpdvdb
  :alt: PyPI
.. |status| image:: https://img.shields.io/pypi/status/kpdvdb
  :alt: PyPI - Status
.. |pyver| image:: https://img.shields.io/pypi/pyversions/kpdvdb
  :alt: PyPI - Python Version
.. |license| image:: https://img.shields.io/github/license/tikuma-lsuhsc/python-kpdvdb
  :alt: GitHub

This Python module provides functions to interact with KayPENTAX Disordered Voice Database

This module DOES NOT provide the database itself. KayPENTAX Disordered Voice Database is 
a (discontinued) commercial product, an addon option to KayPENTAX's Computerized Speech Lab (CSL).

For faster access to the data, copying the database files from the CD-ROM to a local hard drive is 
highly recommended.

Install
-------

.. code-block:: bash

  pip install kpdvdb

Use
---

.. code-block:: python

  import kpdvdb

  # to read docstrings
  help(kpdvdb)

  # to initialize (must call this once in every Python session)
  kpdvdb.load_db('<path to CDROM drive or root directory of the database>')

  # to list all the data fields 
  print(kpdvdb.get_fields())

  # to list categorical fields' unique values
  print(kpdvdb.get_sexes()) # genders
  print(kpdvdb.get_locations()) # pathology sites
  print(kpdvdb.get_natlangs()) # native languages
  print(kpdvdb.get_origins()) # races
  print(kpdvdb.get_diagnoses()) # diagnoses

  # to get a copy of the full database
  df = kpdvdb.query(include_diagnoses=True)

  # to get age, gender, diagnoses, and MDVP measures of non-smoking 
  # subjects with polyp or paralysis, F0 between 100 and 300 Hz
  df = kpdvdb.query(["AGE","SEX","DIAGNOSES","MDVP"], 
                    DIAGNOSES=["vocal fold polyp","paralysis"],
                    Fo=[100,300],
                    SMOKE=False)

  # to get the list of AH NSP files of normal subjects
  nspfiles = kpdvdb.get_files('ah',NORM=True)

  # to iterate over 'rainbow passage' acoustic data of female pathological subjects
  for fs, x, info in kpdvdb.iter_data('rainbow',
                                      auxdata_fields=["AGE","SEX"],
                                      NORM=False, SEX="F"):
    # run the acoustic data through your analysis function, get measurements
    params = my_analysis_function(fs, x)

    # log the measurements along with the age and gender info
    my_logger.log_outcome(*info, *params)

NOTE
----

Because the database is not public, this module cannot be tested for various platforms
via GitHub Action. If you encounter any issues, please post it on GitHub.
