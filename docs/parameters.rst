Parameters
==========

This page describes the Python parameter dictionary used by ``SingleOutputGBDT`` and ``MultiOutputGBDT`` in this fork. Unless noted otherwise, defaults come from ``omnigbdt.lib_utils.default_params()``.

General
-------

- ``loss``: default = ``b"mse"``, type = bytes
  - Supported values are ``b"mse"``, ``b"bce"``, ``b"ce"``, and ``b"ce_column"``
  - ``b"ce_column"`` is only relevant to legacy ``SingleOutputGBDT`` classification-style workflows
  - The Python API expects a byte string, for example ``b"mse"``

- ``verbosity``: default = ``Verbosity.FULL`` (``2``), type = ``Verbosity`` or int
  - ``Verbosity.SILENT`` / ``0`` prints nothing from the native trainer
  - ``Verbosity.SUMMARY`` / ``1`` prints only the final best score when evaluation data is present
  - ``Verbosity.FULL`` / ``2`` prints per-round metrics and the final best score

- ``verbose``: default = ``True``, type = bool
  - Backward-compatible alias for the old two-level behavior
  - ``False`` maps to ``Verbosity.SILENT``
  - ``True`` maps to ``Verbosity.FULL``

- ``num_threads``: default = ``2``, type = int
  - Number of training threads

- ``seed``: default = ``0``, type = int
  - Random seed
  - The upstream code notes that this currently has limited practical effect

- ``hist_cache``: default = ``16``, type = int
  - Maximum number of histogram caches

- ``max_bins``: default = ``32``, type = int
  - Maximum number of bins for each input feature

- ``topk``: default = ``0``, type = int
  - Sparse split-finding parameter
  - If ``0``, the dense split-search path is used

- ``one_side``: default = ``True``, type = bool
  - Selects the sparse split-search variant
  - Only used when ``topk != 0``

Tree
----

- ``max_depth``: default = ``4``, type = int
  - Maximum tree depth
  - Must be at least ``1``

- ``max_leaves``: default = ``32``, type = int
  - Maximum number of leaves per tree

- ``min_samples``: default = ``20``, type = int
  - Minimum number of samples allowed in a leaf

- ``early_stop``: default = ``0``, type = int
  - Early-stopping patience in rounds
  - If ``0``, early stopping is disabled

Learning
--------

- ``lr``: default = ``0.2``, type = float
  - Learning rate

- ``base_score``: default = ``0.0``, type = float
  - Initial prediction value

- ``reg_l1``: default = ``0.0``, type = float
  - L1 regularization term
  - The upstream code notes that this is not currently used for sparse split finding

- ``reg_l2``: default = ``1.0``, type = float
  - L2 regularization term

- ``gamma``: default = ``1e-3``, type = float
  - Minimum objective gain required for a split

- ``subsample``: default = ``1.0``, type = float
  - Present in the Python defaults for compatibility
  - The current native implementation does not actively use it

Model-specific notes
--------------------

- ``MultiOutputGBDT`` expects multi-output labels shaped like ``(n_samples, out_dim)``
- ``SingleOutputGBDT`` is best used with one target column at a time
- if you want a simple multi-output baseline with ``SingleOutputGBDT``, train one model per target column and stack their predictions manually
- ``SingleOutputGBDT.train_multi(...)`` is a legacy helper for multi-class classification style workflows, not the common baseline path used in this fork's examples
