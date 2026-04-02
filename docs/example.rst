Examples
========

This page contains short, self-contained examples for the packaged OmniGBDT fork. For benchmark scripts and extended evaluation workflows, please see the upstream repositories:

- `GBDTMO <https://github.com/zzd1992/GBDTMO>`_
- `GBDTMO-EX <https://github.com/zzd1992/GBDTMO-EX>`_

Basic multi-output training
---------------------------

.. code-block:: python

   import numpy as np
   from omnigbdt import MultiOutputGBDT, Verbosity

   rng = np.random.default_rng(0)

   X = rng.random((512, 4)).astype("float64")
   shared_signal = (
       1.5 * X[:, 0]
       - 0.8 * X[:, 1]
       + 0.4 * np.sin(np.pi * X[:, 2])
   )
   Y = np.column_stack([
       1.2 * shared_signal + 0.3 * X[:, 2] * X[:, 3],
       0.9 * shared_signal - 0.4 * X[:, 0] + 0.2 * X[:, 3],
       1.1 * shared_signal + 0.5 * X[:, 1] * X[:, 3],
   ]).astype("float64")
   Y += 0.05 * rng.standard_normal(512)[:, None]
   Y += 0.02 * rng.standard_normal((512, 3))

   params = {
       "loss": b"mse",
       "max_depth": 3,
       "lr": 0.1,
       "num_threads": 1,
       "verbosity": Verbosity.SILENT,
   }

   booster = MultiOutputGBDT(out_dim=Y.shape[1], params=params)
   booster.set_data((X, Y))
   booster.train(1)

   preds = booster.predict(X[:5])
   print(preds.shape)

Comparing ``SingleOutputGBDT`` and ``MultiOutputGBDT``
------------------------------------------------------

One simple baseline is to train:

- one ``MultiOutputGBDT`` model on the full target matrix
- one ``SingleOutputGBDT`` model per target column

.. code-block:: python

   import numpy as np
   from omnigbdt import SingleOutputGBDT, MultiOutputGBDT, Verbosity

   rng = np.random.default_rng(0)

   n_samples = 512
   n_features = 4
   n_outputs = 3

   X = rng.random((n_samples, n_features)).astype("float64")
   shared_signal = (
       1.5 * X[:, 0]
       - 0.8 * X[:, 1]
       + 0.4 * np.sin(np.pi * X[:, 2])
   )
   target_specific = np.column_stack([
       0.3 * X[:, 2] * X[:, 3],
       -0.4 * X[:, 0] + 0.2 * X[:, 3],
       0.5 * X[:, 1] * X[:, 3],
   ])
   shared_noise = 0.05 * rng.standard_normal(n_samples)[:, None]
   independent_noise = 0.02 * rng.standard_normal((n_samples, n_outputs))
   Y = np.column_stack([
       1.2 * shared_signal,
       0.9 * shared_signal,
       1.1 * shared_signal,
   ]).astype("float64")
   Y += target_specific + shared_noise + independent_noise

   params = {
       "loss": b"mse",
       "max_depth": 3,
       "lr": 0.1,
       "num_threads": 1,
       "verbosity": Verbosity.SILENT,
   }

   multi = MultiOutputGBDT(out_dim=n_outputs, params=params)
   multi.set_data((X, Y))
   multi.train(1)
   multi_preds = multi.predict(X)

   single_models = []
   for col in range(n_outputs):
       model = SingleOutputGBDT(params=params)
       target = np.ascontiguousarray(Y[:, col])
       model.set_data((X, target))
       model.train(1)
       single_models.append(model)

   single_preds = np.column_stack([model.predict(X) for model in single_models])

   multi_rmse = np.sqrt(np.mean((multi_preds - Y) ** 2))
   single_rmse = np.sqrt(np.mean((single_preds - Y) ** 2))

   print("MultiOutputGBDT RMSE:", round(float(multi_rmse), 6))
   print("SingleOutputGBDT-per-target RMSE:", round(float(single_rmse), 6))

Dumping and loading a model
---------------------------

Continuing from the basic multi-output example above:

.. code-block:: python

   from pathlib import Path

   model_path = Path("omnigbdt_model.txt")
   booster.dump(model_path)

   reloaded = MultiOutputGBDT(out_dim=Y.shape[1], params=params)
   reloaded.set_booster(X.shape[1], Y.shape[1])
   reloaded.load(model_path)

Advanced manual loading
-----------------------

Normal usage does not require manual shared-library handling, but the compatibility helper is still available:

.. code-block:: python

   from omnigbdt import MultiOutputGBDT, load_lib

   lib = load_lib("/path/to/native/library/or/folder")
   booster = MultiOutputGBDT(lib=lib, out_dim=3, params={"loss": b"mse"})

Optional plotting
-----------------

Install the optional plotting dependency first:

.. code-block:: bash

   pip install "omnigbdt[plot]"

Then render a dumped tree:

.. code-block:: python

   from omnigbdt import create_graph

   graph = create_graph("omnigbdt_model.txt", tree_index=0, value_list=[0, 1])
   graph.render("tree_0", format="pdf")

Custom loss
-----------

``MultiOutputGBDT`` supports public callback-based custom objectives through ``train(..., objective=...)``:

Continuing from the basic multi-output example above:

.. code-block:: python

   import numpy as np

   def mse_objective(preds, target):
       return preds - target, np.ones_like(preds)

   def rmse_metric(preds, target):
       return float(np.sqrt(np.mean((preds - target) ** 2)))

   booster.train(
       10,
       objective=mse_objective,
       eval_metric=rmse_metric,
       maximize=False,
   )

This uses your Python callback to supply gradients and Hessians round by round.

If you need manual control, the protected ``_set_gh(...)`` plus ``boost()`` workflow still exists as an advanced escape hatch:

.. code-block:: python

   g, h = mse_objective(booster.preds_train.copy(), booster.label.copy())
   booster._set_gh(g, h)
   booster.boost()

For ``SingleOutputGBDT``, the custom-objective callback receives 1D arrays. For ``MultiOutputGBDT``, it receives 2D arrays shaped ``(n_samples, out_dim)``.

The sklearn-compatible wrappers forward the same callback arguments:

.. code-block:: python

   from omnigbdt import MultiOutputGBDTRegressor

   model = MultiOutputGBDTRegressor(
       num_rounds=10,
       objective=mse_objective,
       eval_metric=rmse_metric,
       maximize=False,
       max_depth=3,
       num_threads=1,
   )
   model.fit(X, Y)

Permutation importance with sklearn
-----------------------------------

Install the optional sklearn extra first:

.. code-block:: bash

   pip install "omnigbdt[sklearn]"

Then use the sklearn-compatible wrapper with ``permutation_importance``:

.. code-block:: python

   import numpy as np
   from sklearn.inspection import permutation_importance

   from omnigbdt import MultiOutputGBDTRegressor

   rng = np.random.default_rng(0)
   X = rng.random((256, 4)).astype("float64")
   shared_signal = (
       1.2 * X[:, 0]
       - 0.7 * X[:, 1]
       + 0.3 * np.sin(np.pi * X[:, 2])
   )
   Y = np.column_stack([
       1.1 * shared_signal + 0.2 * X[:, 3],
       0.9 * shared_signal - 0.3 * X[:, 0],
       1.0 * shared_signal + 0.4 * X[:, 1] * X[:, 3],
   ]).astype("float64")
   Y += 0.05 * rng.standard_normal(256)[:, None]
   Y += 0.02 * rng.standard_normal((256, 3))

   model = MultiOutputGBDTRegressor(
       num_rounds=10,
       max_depth=3,
       num_threads=1,
   )
   model.fit(X, Y)

   result = permutation_importance(
       model,
       X,
       Y,
       scoring="r2",
       n_repeats=5,
       random_state=42,
       n_jobs=1,
   )

   print(result.importances_mean)
