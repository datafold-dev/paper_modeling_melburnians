Supplementary material: Modeling Melburnians - Using the Koopman operator to gain insight into crowd dynamic
============================================================================================================

Code and data source
--------------------

The core model implementations are performed with our own Python package *datafold*

* `Documentation page <https://datafold-dev.gitlab.io/datafold/>`__
* `Paper published in JOSS <https://joss.theoj.org/papers/10.21105/joss.02283>`__

The primary source of the data included in this repository is provided by the city of Melbourne:

* `Pedestrian Counting System - Monthly (counts per hour) <https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Monthly-counts-per-hour/b2ak-trbp>`__
* `Pedestrian Counting System - Sensor Locations <https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Sensor-Locations/h57g-5234>`__
* `City map with recent data <http://www.pedestrian.melbourne.vic.gov.au/>`__

The data is licensed under the Creative Commons Attribution 4.0 International Public
License. For details see:
https://creativecommons.org/licenses/by/4.0/
https://creativecommons.org/licenses/by/4.0/legalcode


Mirror of Supplementary Material
--------------------------------

A mirror of the supplementary material which also contains cached ``.csv`` file is located at
`github <https://github.com/datafold-dev/paper_modeling_melburnians>`__.

Because the data is versioned with git-lfs (Git Large File Storage), it can only be cloned
via git:

.. code-block::

    git clone git@github.com:datafold-dev/paper_modeling_melburnians.git

(A download of the repository in a ``.zip`` file only contains file links to the data.)

Run notebook
------------

Before running ``notebook.ipynb``, note that the computations can take a while, depending
on your hardware. A minimum of 16 GiB memory is required. However, if intermediate results
are cached in csv files and the flag ``use_cache`` is enabled in the notebook , most
computational expensive operations are avoided.

The ``notebook.ipynb`` reproduces and visualizes results from the paper. The file
``main.py`` includes the code for training the EDMD model and plotting.

Jupyter notebook
^^^^^^^^^^^^^^^^

To execute the code `Python>=3.7 <https://www.python.org/>`__ and
`Jupyter <https://jupyter.org/>`__ is required.

To install the package requirements run:

.. code-block::

    pip install datafold==1.1.4 holidays==0.10.4

Open the Jupyter notebook with

.. code-block::

    jupyter notebook notebook.ipynb


Overview of data files:
^^^^^^^^^^^^^^^^^^^^^^^

**Raw data:**

* ``X_all.csv`` -- raw data of the original source with all sensors between 1/1/2016 -
  12/31/2019 with sampling rate of 1 hour

**Cached data:**

Note, that because of size restrictions in the official supplementary material of the
paper, the cached files are only included in the github repository (see link above). All
cache files are generated locally, if the notebook executes runs with ``use_cache=False``.

* ``X_selected.csv``
   selected sensors and samples from ``X_all.csv`` as highlighted in the paper

* ``X_windows_[train|test].csv``
   data of time series windows of length 193; these are used for the time series pairs with
   169 hours (initial condition) + 24 hours (prediction)

* ``X_reconstruct_[train|test].csv``
   reconstructed data in time series of length 24 for each window in corresponding
   ``X_windows``

* ``X_latent_[train|test].csv``
   diffusion map values for each prediction window

* ``X_latent_interp_test.csv``
   interpolated diffusion map values with EDMD model

* ``X_eigfunc_test.csv``
   complex Koopman eigenfunction values
