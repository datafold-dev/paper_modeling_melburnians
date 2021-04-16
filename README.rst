
Supplementary material: Modeling Melburnians - Using the Koopman operator to gain insight into crowd dynamic
============================================================================================================

Associated paper:
-----------------

[TODO: INSERT_LINK]
[TODO: INSERT_CITATION_TEXT]

Code and data source
--------------------

The core model implementations are performed with our own Python package *datafold*

* `Documentation <https://datafold-dev.gitlab.io/datafold/>`__
* `Paper published in JOSS <https://joss.theoj.org/papers/10.21105/joss.02283>`__

The primary source of the data included in this repository is provided by the city of Melbourne:

* `Pedestrian Counting System - Monthly (counts per hour) <https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Monthly-counts-per-hour/b2ak-trbp>`__
* `Pedestrian Counting System - Sensor Locations <https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Sensor-Locations/h57g-5234>`__
* `City map with recent data <http://www.pedestrian.melbourne.vic.gov.au/>`__

Mirror of Supplementary Material
--------------------------------

A mirror of the supplementary material which also hosts the cached file is located at
[INSERT_GITHUB|GITLAB_LINK].

Cite as
-------

[TODO: INSERT_CITATION]


Run notebook
------------

Note that the computations can take a while, depending on your hardware.
A minimum of 16 GiB memory is required. However, if intermediate results are cached in
csv files and the flag ``use_cache`` in the notebook is enabled, the computational
expensive operations are disabled.

Jupyter notebook
^^^^^^^^^^^^^^^^

The code requires ``Python>=3.7` <https://www.python.org/>`__ and `Jupyter <https://jupyter.org/>`__ to be installed.

To install the package requirements run:

.. code-block::

    pip install datafold==1.1.4 holidays==0.10.4

Open the Jupyter notebook with

.. code-block::

    jupyter notebook notebook.ipynb


Overview of all files:
^^^^^^^^^^^^^^^^^^^^^^

**Raw data:**

* ``X_all.csv`` -- raw data of all sensors between 1/1/2016 - 12/31/2019

**Cached data:**

Note, that these are only included in the github|gitlab mirror, because of size restrictions in the official supplementary material.

* ``X_selected.csv`` -- selected sensors and samples in ``X_all.csv``
* ``X_windows_[train|test].csv`` -- data in time series of length 192; 168 hour (initial condition) + 24 hours (prediction)
* ``X_reconstruct_[train|test].csv`` -- reconstructed data in time series of length 24 for each window in corresponding ``X_windows``
* ``X_latent_[train|test].csv`` -- diffusion map values for each prediction window
* ``X_latent_interp_test.csv`` -- with EDMD model interpolated diffusion map values
* ``X_eigfunc_test.csv`` -- complex Koopman eigenfunction values
