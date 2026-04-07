Installation
============

This page covers released-package installs, optional extras, source installs, and local-path workflows.

Released package
----------------

Install the released package:

.. code-block:: bash

   pip install omnigbdt

or with ``uv``:

.. code-block:: bash

   uv add omnigbdt

The current wheel targets are:

- Linux x86_64
- Windows x86_64
- macOS arm64 (Apple Silicon, 14+)

Optional extras
---------------

Optional extras are available for plotting and sklearn-compatible wrappers:

.. code-block:: bash

   pip install "omnigbdt[plot]"
   pip install "omnigbdt[sklearn]"

or with ``uv``:

.. code-block:: bash

   uv add "omnigbdt[plot]"
   uv add "omnigbdt[sklearn]"

The optional sklearn wrappers are a fork-specific addition. They make it possible to use sklearn inspection utilities such as permutation-based feature importance:

- `Feature importance based on feature permutation <https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-feature-permutation>`_
- `Permutation feature importance <https://scikit-learn.org/stable/modules/permutation_importance.html>`_

Install from source
-------------------

Install from a source checkout:

.. code-block:: bash

   pip install .

or with ``uv`` from a parent directory:

.. code-block:: bash

   uv add ./OmniGBDT

On Windows, use either ``uv add .\\OmniGBDT`` or ``uv add ./OmniGBDT``.
Do not use ``uv add OmniGBDT`` without ``./`` or ``.\\`` because that requests the published registry package instead of the local folder.

Use OmniGBDT inside an existing uv project
------------------------------------------

Add OmniGBDT as a normal released dependency:

.. code-block:: bash

   uv add omnigbdt

Add a sibling checkout as an editable dependency:

.. code-block:: bash

   uv add --editable ../OmniGBDT

If the ``OmniGBDT`` folder is copied inside an existing ``uv`` workspace and the command below is used:

.. code-block:: bash

   uv add ./OmniGBDT

then ``uv`` may treat it as a workspace member. If a plain path dependency is preferred instead, use:

.. code-block:: bash

   uv add --no-workspace ./OmniGBDT

The equivalent manual configuration in ``pyproject.toml`` is:

.. code-block:: toml

   [project]
   dependencies = ["omnigbdt"]

   [tool.uv.sources]
   omnigbdt = { path = "../OmniGBDT", editable = true }

Windows source builds
---------------------

Local path installs such as ``uv add ./OmniGBDT`` and source installs such as ``pip install .`` compile the native C++ library during installation.

On Windows, install Visual Studio Build Tools 2022 or Visual Studio 2022 with the ``Desktop development with C++`` workload before building from source.

If CMake reports that ``nmake`` is missing and ``CMAKE_CXX_COMPILER`` is not set, the current shell does not have a usable MSVC toolchain configured. Reopen the terminal from ``x64 Native Tools Command Prompt for VS 2022`` and try the install again. If the toolchain is already installed, check that ``CMAKE_GENERATOR`` is not forcing ``NMake Makefiles`` in a shell where ``nmake.exe`` is unavailable.
