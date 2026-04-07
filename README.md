# OmniGBDT

OmniGBDT packages the original [GBDT-MO](https://github.com/zzd1992/GBDTMO) algorithm as a regular Python library. It keeps the native C++ training core and adds modern Python packaging, cross-platform wheels, public custom-objective hooks, and optional sklearn-compatible wrappers.

The main public classes are `MultiOutputGBDT` and `SingleOutputGBDT`.

For the original project, benchmark figures, experiment scripts, and upstream research context, please see:

- Original repository: <https://github.com/zzd1992/GBDTMO>
- Experiment and evaluation repository: <https://github.com/zzd1992/GBDTMO-EX>

## Installation

### Install the released package

```bash
pip install omnigbdt
```

or with `uv`:

```bash
uv add omnigbdt
```

OmniGBDT targets wheel-based installs on:

- Linux x86_64
- Windows x86_64
- macOS arm64 (Apple Silicon, 14+)

The GitHub Actions workflow builds these wheels in CI and publishes them on version tags matching `v*`.

### Optional extras

Install plotting support if you want to render trees with `create_graph()`:

```bash
pip install "omnigbdt[plot]"
```

Install sklearn-compatible wrappers if you want to use `permutation_importance`:

```bash
pip install "omnigbdt[sklearn]"
```

The same extras can be installed with `uv`:

```bash
uv add "omnigbdt[plot]"
uv add "omnigbdt[sklearn]"
```

The optional sklearn wrappers are a fork-specific addition. They make it possible to use sklearn inspection utilities such as permutation-based feature importance:

- [Feature importances with a forest of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-feature-permutation)
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

## Quick Start

### Finance benchmark example

Install the UCI dataset helper used in this example:

```bash
pip install ucimlrepo
```

or with `uv`:

```bash
uv add ucimlrepo
```

The example below uses the [UCI Machine Learning Repository Stock Portfolio Performance dataset](https://archive.ics.uci.edu/dataset/390/stock+portfolio+performance), comparing:

- a `MultiOutputGBDT` model trained on the full target matrix
- a `SingleOutputGBDT` model per target column

```python
import numpy as np
from ucimlrepo import fetch_ucirepo

from omnigbdt import SingleOutputGBDT, MultiOutputGBDT, Verbosity

stock_portfolio = fetch_ucirepo(id=390)
frame = stock_portfolio.data.original

feature_columns = [
    "Large B/P",
    "Large ROE",
    "Large S/P",
    "Large Return Rate in the last quarter",
    "Large Market Value",
    "Small systematic Risk",
]
target_columns = [
    "Annual Return.1",
    "Excess Return.1",
    "Systematic Risk.1",
    "Total Risk.1",
    "Abs. Win Rate.1",
    "Rel. Win Rate.1",
]

X = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
Y = frame.loc[:, target_columns].to_numpy(dtype=np.float64)

rng = np.random.default_rng(0)
indices = rng.permutation(len(X))
train_end = int(len(X) * 0.6)
valid_end = int(len(X) * 0.8)
train_idx = indices[:train_end]
valid_idx = indices[train_end:valid_end]
test_idx = indices[valid_end:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_valid, Y_valid = X[valid_idx], Y[valid_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

params = {
    "loss": b"mse",
    "max_depth": 4,
    "max_bins": 128,
    "lr": 0.05,
    "early_stop": 15,
    "num_threads": 1,
    "verbosity": Verbosity.SILENT,
}

multi = MultiOutputGBDT(out_dim=Y.shape[1], params=params)
multi.set_data((X_train, Y_train), (X_valid, Y_valid))
multi.train(200)
multi_preds = multi.predict(X_test)

single_models = []
for col in range(Y.shape[1]):
    model = SingleOutputGBDT(params=params)
    target = np.ascontiguousarray(Y_train[:, col])
    eval_target = np.ascontiguousarray(Y_valid[:, col])
    model.set_data((X_train, target), (X_valid, eval_target))
    model.train(200)
    single_models.append(model)

single_preds = np.column_stack([model.predict(X_test) for model in single_models])

multi_rmse = np.sqrt(np.mean((multi_preds - Y_test) ** 2))
single_rmse = np.sqrt(np.mean((single_preds - Y_test) ** 2))

print("Held-out RMSE from MultiOutputGBDT:", round(float(multi_rmse), 6))
print("Held-out RMSE from stacked SingleOutputGBDT models:", round(float(single_rmse), 6))
print("Prediction shape from MultiOutputGBDT:", multi.predict(X_test[:3]).shape)
print("Prediction shape from stacked SingleOutputGBDT models:", single_preds[:3].shape)
```

The UCI export includes both formatted percentage columns and normalized numeric target columns. The example above uses the normalized target columns from `data.original`, which carry the `.1` suffix in the spreadsheet-derived column names. `base_score` is left unset, so regression training starts from the training-label mean automatically.

### Custom objectives

Continuing from the stock portfolio split above, gradients and Hessians can be supplied from Python with a custom objective callback:

```python
import numpy as np

from omnigbdt import MultiOutputGBDT, Verbosity


def mse_objective(preds, target):
    return preds - target, np.ones_like(preds)


def rmse_metric(preds, target):
    return float(np.sqrt(np.mean((preds - target) ** 2)))


booster = MultiOutputGBDT(
    out_dim=Y_train.shape[1],
    params={
        "loss": b"mse",
        "max_depth": 4,
        "max_bins": 128,
        "lr": 0.05,
        "early_stop": 15,
        "num_threads": 1,
        "verbosity": Verbosity.FULL,
    },
)
booster.set_data((X_train, Y_train), (X_valid, Y_valid))
booster.train(
    200,
    objective=mse_objective,
    eval_metric=rmse_metric,
    maximize=False,
)
preds = booster.predict(X_valid)
print(preds.shape)
```

Notes:

- `SingleOutputGBDT.train(..., objective=...)` expects 1D `preds` and `target` arrays.
- `MultiOutputGBDT.train(..., objective=...)` expects 2D arrays shaped `(n_samples, out_dim)`.
- `loss` must still be a supported built-in native loss name such as `b"mse"` because the native booster validates it at construction time, but custom rounds use the Python callback instead of the built-in objective.
- If `early_stop > 0` and evaluation labels are registered on the custom-objective path, `eval_metric` and `maximize` are also required.
- The protected `_set_gh(...)` plus `boost()` workflow still exists as an advanced manual escape hatch.

### Permutation importance with sklearn

Install the optional sklearn extra and the UCI dataset helper first:

```bash
pip install "omnigbdt[sklearn]"
pip install ucimlrepo
```

or with `uv`:

```bash
uv add "omnigbdt[sklearn]"
uv add ucimlrepo
```

Then use the sklearn-compatible multi-output wrapper with permutation importance:

```python
import time

import numpy as np
from sklearn.inspection import permutation_importance
from ucimlrepo import fetch_ucirepo

from omnigbdt import MultiOutputGBDTRegressor

stock_portfolio = fetch_ucirepo(id=390)
frame = stock_portfolio.data.original
feature_columns = [
    "Large B/P",
    "Large ROE",
    "Large S/P",
    "Large Return Rate in the last quarter",
    "Large Market Value",
    "Small systematic Risk",
]
target_columns = [
    "Annual Return.1",
    "Excess Return.1",
    "Systematic Risk.1",
    "Total Risk.1",
    "Abs. Win Rate.1",
    "Rel. Win Rate.1",
]
X = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
Y = frame.loc[:, target_columns].to_numpy(dtype=np.float64)

rng = np.random.default_rng(0)
indices = rng.permutation(len(X))
train_end = int(len(X) * 0.8)
train_idx = indices[:train_end]
test_idx = indices[train_end:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

model = MultiOutputGBDTRegressor(
    num_rounds=200,
    max_depth=4,
    max_bins=128,
    lr=0.05,
    early_stop=15,
    num_threads=1,
)
model.fit(X_train, Y_train)

start_time = time.time()
result = permutation_importance(
    model,
    X_test,
    Y_test,
    scoring="r2",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)
elapsed_time = time.time() - start_time

print(f"Elapsed time: {elapsed_time:.3f} seconds")
print(result.importances_mean)
```

The sklearn-compatible wrappers also accept `objective=...`, `eval_metric=...`, and `maximize=...`, and forward them to the same custom-objective training path.

## Source and Development Installs

### Install from source

```bash
pip install .
```

or with `uv`:

```bash
uv add ./OmniGBDT
```

On Windows, use either `uv add .\\OmniGBDT` or `uv add ./OmniGBDT`.
Do not use `uv add OmniGBDT` without `./` or `.\\`, because that asks the package registry for a published package named `omnigbdt` instead of using the local folder.

### Use OmniGBDT inside an existing uv project

Add OmniGBDT as a normal released dependency:

```bash
uv add omnigbdt
```

Add a sibling checkout as an editable dependency while developing two projects side by side:

```bash
uv add --editable ../OmniGBDT
```

If you copy the `OmniGBDT` folder inside an existing `uv` workspace and run:

```bash
uv add ./OmniGBDT
```

then `uv` may treat it as a workspace member. If you want it to remain a plain path dependency, use:

```bash
uv add --no-workspace ./OmniGBDT
```

On Windows, the same commands are:

```bash
uv add .\\OmniGBDT
uv add --no-workspace .\\OmniGBDT
```

The equivalent manual configuration in `pyproject.toml` is:

```toml
[project]
dependencies = ["omnigbdt"]

[tool.uv.sources]
omnigbdt = { path = "../OmniGBDT", editable = true }
```

### Windows source builds

Local installs (`uv add ./OmniGBDT` or `pip install .`) compile the native C++ library during installation.

So on Windows, you must install first:

- Visual Studio Build Tools 2022 (or Visual Studio 2022)
- the `Desktop development with C++` workload
- MSVC build tools and a working OpenMP-capable compiler

If CMake fails with an error such as:

```text
Running 'nmake' '-?' failed with: no such file or directory
CMAKE_CXX_COMPILER not set, after EnableLanguage
```

then the package is being built from source but the MSVC toolchain is not available in the current shell.

In this case, try:

1. Installing Visual Studio Build Tools 2022 with the C++ workload.
2. Reopening the terminal from `x64 Native Tools Command Prompt for VS 2022`.
3. Rerun `uv add ./OmniGBDT` or `pip install .`.

If the toolchain is already installed, also check that `CMAKE_GENERATOR` is not forcing `NMake Makefiles` in a shell where `nmake.exe` is unavailable.

## What OmniGBDT Adds

Compared with the upstream GBDTMO repository, OmniGBDT:

- replaces the old `make.sh` and manual shared-library workflow with standard Python packaging
- bundles the native library inside the Python package
- keeps `load_lib(path=None)` for advanced or compatibility workflows
- adds wheel automation for Linux, macOS, and Windows
- adds public Python callback hooks for custom gradients, Hessians, metrics, and early stopping
- adds optional sklearn-compatible wrappers so users can apply sklearn inspection tools such as permutation-based feature importance

### Core native-code deviations from upstream GBDT-MO

Most changes in this fork are packaging and distribution changes. The native C/C++ training code has only been changed in a few targeted ways so far:

- stricter `min_samples` enforcement during split scoring: candidate split points are rejected unless both child branches satisfy `min_samples`
- safe child-node materialization after a split: if a branch cannot be split further, it is emitted as an explicit leaf instead of being left implicit or partially unassigned
- proper root-leaf fallback: if no valid split exists at the root, the model stores a true single-leaf tree and prediction, dump, and load work cleanly for that case

As a consequence, same-seed runs do not necessarily match older buggy runs exactly. Trees can be smaller because invalid small-child splits are filtered earlier, and the control flow through the native code changes accordingly.

Outside of those fixes, the core objective functions, histogram-based split search, and overall training structure are still inherited from the original repository.

## Project Provenance

This fork builds directly on the original GBDT-MO implementation by Zhendong Zhang and Cheolkon Jung.

OmniGBDT is intended to make the package easier to build, install, and distribute. It is not the canonical source for the paper, benchmark tables, figures, or research documentation.

For evaluation metrics, dataset-specific experiments, and extended project context, please refer to:

- <https://github.com/zzd1992/GBDTMO>
- <https://github.com/zzd1992/GBDTMO-EX>

## Development

### Run tests

For local development with `uv`, sync the project together with the optional test dependencies:

```bash
uv sync --extra test
```

Then run the test suite with:

```bash
uv run pytest
```

### Build the documentation locally

The hosted documentation is configured through the repository-level `.readthedocs.yaml` file and the pinned dependencies in `docs/requirements.txt`.

To preview the docs locally with the same Sphinx dependency set used on Read the Docs, run:

```bash
uv run --no-project --with-requirements docs/requirements.txt sphinx-build -W -n -b html docs _build/html
```

### Build the native library directly

To build the native library:

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Versioning

This fork follows Semantic Versioning independently from the upstream GBDT-MO repository.

## License

This fork is distributed under the Apache License 2.0. The main license text for this fork is in [LICENSE](LICENSE).

Because this repository incorporates and modifies the original GBDT-MO codebase, the original upstream MIT license notice from Zhendong Zhang is preserved in [LICENSE.upstream](LICENSE.upstream). Additional attribution and fork-specific notice text is provided in [NOTICE](NOTICE).

## Citation

If you use this project in research, please credit the original paper by Zhang and Jung:

```bibtex
@article{zhang2020gbdt,
  title={GBDT-MO: Gradient-boosted decision trees for multiple outputs},
  author={Zhang, Zhendong and Jung, Cheolkon},
  journal={IEEE transactions on neural networks and learning systems},
  volume={32},
  number={7},
  pages={3156--3167},
  year={2020},
  publisher={Ieee}
}
```
