Development
===========

This page collects local contributor workflows that are useful during package development.

Run tests
---------

For local development with ``uv``, sync the project together with the optional test dependencies:

.. code-block:: bash

   uv sync --extra test

Then run the test suite:

.. code-block:: bash

   uv run pytest

Build the documentation locally
-------------------------------

The hosted documentation is configured through the repository-level ``.readthedocs.yaml`` file and the pinned dependencies in ``docs/requirements.txt``.

To preview the docs locally with the same Sphinx dependency set used on Read the Docs, run:

.. code-block:: bash

   uv run --no-project --with-requirements docs/requirements.txt sphinx-build -W -n -b html docs _build/html

Build the native library directly
---------------------------------

To build the native library without reinstalling the package:

.. code-block:: bash

   cmake -S . -B build
   cmake --build build --config Release
