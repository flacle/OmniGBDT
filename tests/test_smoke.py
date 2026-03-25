from pathlib import Path
import importlib.util

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT, Verbosity, load_lib
from omnigbdt.lib_utils import _resolve_packaged_library_path


def test_multioutputgbdt_smoke(tmp_path):
    rng = np.random.default_rng(0)
    inp_dim = 10
    out_dim = 5
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "num_threads": 1,
        "verbose": False,
    }

    x_train = rng.random((512, inp_dim)).astype("float64")
    y_train = rng.random((512, out_dim)).astype("float64")
    x_valid = rng.random((128, inp_dim)).astype("float64")
    y_valid = rng.random((128, out_dim)).astype("float64")

    booster = MultiOutputGBDT(out_dim=out_dim, params=params)
    booster.set_data((x_train, y_train), (x_valid, y_valid))
    booster.train(1)
    preds = booster.predict(x_valid)

    assert preds.shape == (len(x_valid), out_dim)

    model_path = tmp_path / "model.txt"
    booster.dump(model_path)
    assert model_path.is_file()

    loaded = MultiOutputGBDT(out_dim=out_dim, params=params)
    loaded.set_booster(inp_dim, out_dim)
    loaded.load(model_path)
    loaded_preds = loaded.predict(x_valid)

    np.testing.assert_allclose(preds, loaded_preds)

    explicit_lib = load_lib(str(_resolve_packaged_library_path()))
    assert explicit_lib is booster.lib

    booster.close()
    loaded.close()


def test_multioutputgbdt_verbosity_levels(capfd):
    rng = np.random.default_rng(1)
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 2,
        "num_threads": 1,
    }

    x_train = rng.random((64, 4)).astype("float64")
    y_train = rng.random((64, 2)).astype("float64")
    x_valid = rng.random((32, 4)).astype("float64")
    y_valid = rng.random((32, 2)).astype("float64")

    silent = MultiOutputGBDT(out_dim=2, params={**params, "verbosity": Verbosity.SILENT})
    silent.set_data((x_train, y_train), (x_valid, y_valid))
    silent.train(2)
    silent.close()

    silent_out = capfd.readouterr().out
    assert "Best score" not in silent_out
    assert "[0] train->" not in silent_out

    summary = MultiOutputGBDT(out_dim=2, params={**params, "verbosity": Verbosity.SUMMARY})
    summary.set_data((x_train, y_train), (x_valid, y_valid))
    summary.train(2)
    summary.close()

    summary_out = capfd.readouterr().out
    assert "Best score" in summary_out
    assert "[0] train->" not in summary_out

    full = MultiOutputGBDT(out_dim=2, params={**params, "verbosity": Verbosity.FULL})
    full.set_data((x_train, y_train), (x_valid, y_valid))
    full.train(2)
    full.close()

    full_out = capfd.readouterr().out
    assert "Best score" in full_out
    assert "[0] train->" in full_out


def test_legacy_verbose_flag_maps_to_full_and_silent(capfd):
    rng = np.random.default_rng(2)
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 2,
        "num_threads": 1,
    }

    x_train = rng.random((64, 4)).astype("float64")
    y_train = rng.random((64, 2)).astype("float64")
    x_valid = rng.random((32, 4)).astype("float64")
    y_valid = rng.random((32, 2)).astype("float64")

    silent = MultiOutputGBDT(out_dim=2, params={**params, "verbose": False})
    silent.set_data((x_train, y_train), (x_valid, y_valid))
    silent.train(2)
    silent.close()
    assert capfd.readouterr().out == ""

    noisy = MultiOutputGBDT(out_dim=2, params={**params, "verbose": True})
    noisy.set_data((x_train, y_train), (x_valid, y_valid))
    noisy.train(2)
    noisy.close()

    noisy_out = capfd.readouterr().out
    assert "Best score" in noisy_out
    assert "[0] train->" in noisy_out


def test_multioutputgbdt_falls_back_to_root_leaf_when_no_split_meets_min_samples(tmp_path):
    rng = np.random.default_rng(3)
    x_train = rng.random((6, 2)).astype("float64")
    y_train = rng.random((6, 2)).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "max_leaves": 8,
        "min_samples": 4,
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    booster = MultiOutputGBDT(out_dim=2, params=params)
    booster.set_data((x_train, y_train), (x_train, y_train))
    booster.train(1)
    preds = booster.predict(x_train)

    expected = np.repeat(preds[:1], len(x_train), axis=0)
    np.testing.assert_allclose(preds, expected)

    model_path = tmp_path / "multi_root_leaf.txt"
    booster.dump(model_path)
    dump_text = model_path.read_text()
    assert "\t-" not in dump_text

    loaded = MultiOutputGBDT(out_dim=2, params=params)
    loaded.set_booster(x_train.shape[1], 2)
    loaded.load(model_path)
    loaded_preds = loaded.predict(x_train)

    np.testing.assert_allclose(preds, loaded_preds)

    booster.close()
    loaded.close()


def test_singleoutputgbdt_falls_back_to_root_leaf_when_no_split_meets_min_samples(tmp_path):
    rng = np.random.default_rng(4)
    x_train = rng.random((6, 2)).astype("float64")
    y_train = rng.random(6).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "max_leaves": 8,
        "min_samples": 4,
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    booster = SingleOutputGBDT(params=params)
    booster.set_data((x_train, y_train), (x_train, y_train))
    booster.train(1)
    preds = booster.predict(x_train)

    np.testing.assert_allclose(preds, np.repeat(preds[:1], len(x_train)))

    model_path = tmp_path / "single_root_leaf.txt"
    booster.dump(model_path)
    dump_text = model_path.read_text()
    assert "\t-" not in dump_text

    loaded = SingleOutputGBDT(params=params)
    loaded.set_booster(x_train.shape[1])
    loaded.load(model_path)
    loaded_preds = loaded.predict(x_train)

    np.testing.assert_allclose(preds, loaded_preds)

    booster.close()
    loaded.close()


def test_optional_sklearn_wrappers_are_lazy():
    import omnigbdt

    if importlib.util.find_spec("sklearn") is None:
        with pytest.raises(ImportError, match="scikit-learn"):
            getattr(omnigbdt, "SingleOutputGBDTRegressor")
        with pytest.raises(ImportError, match="scikit-learn"):
            getattr(omnigbdt, "MultiOutputGBDTRegressor")
    else:
        from omnigbdt import MultiOutputGBDTRegressor, SingleOutputGBDTRegressor

        assert MultiOutputGBDTRegressor is not None
        assert SingleOutputGBDTRegressor is not None


def test_singleoutput_sklearn_wrapper_supports_permutation_importance():
    pytest.importorskip("sklearn")
    from sklearn.inspection import permutation_importance

    from omnigbdt import SingleOutputGBDTRegressor

    rng = np.random.default_rng(5)
    x_train = rng.random((128, 4)).astype("float64")
    y_train = (x_train[:, 0] * 2.0 - x_train[:, 1]).astype("float64")

    model = SingleOutputGBDTRegressor(
        num_rounds=5,
        max_depth=3,
        num_threads=1,
        verbosity=Verbosity.SILENT,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_train[:8])

    assert preds.shape == (8,)

    result = permutation_importance(
        model,
        x_train,
        y_train,
        scoring="r2",
        n_repeats=2,
        random_state=42,
        n_jobs=1,
    )

    assert result.importances_mean.shape == (x_train.shape[1],)
    model.close()


def test_multioutput_sklearn_wrapper_smoke():
    pytest.importorskip("sklearn")

    from omnigbdt import MultiOutputGBDTRegressor

    rng = np.random.default_rng(6)
    x_train = rng.random((128, 4)).astype("float64")
    y_train = np.column_stack(
        [
            x_train[:, 0] + x_train[:, 1],
            x_train[:, 2] - x_train[:, 3],
        ]
    ).astype("float64")

    model = MultiOutputGBDTRegressor(
        num_rounds=5,
        max_depth=3,
        num_threads=1,
        verbosity=Verbosity.SILENT,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_train[:8])

    assert preds.shape == (8, y_train.shape[1])
    assert np.isfinite(model.score(x_train, y_train))
    model.close()
