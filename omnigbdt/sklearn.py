from __future__ import annotations

import numpy as np

from .lib_utils import Verbosity
from .models import MultiOutputGBDT, SingleOutputGBDT

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.utils.validation import check_is_fitted
except ImportError as exc:
    raise ImportError(
        "omnigbdt.sklearn requires scikit-learn. Install it with "
        "`pip install \"omnigbdt[sklearn]\"`, "
        "`uv add \"omnigbdt[sklearn]\"`, or `uv sync --extra sklearn`."
    ) from exc


def _as_feature_matrix(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
    return x


def _as_single_target(y):
    y = np.asarray(y)
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(
                "SingleOutputGBDTRegressor expects a 1D target array. "
                "Use MultiOutputGBDTRegressor for multi-output targets."
            )
        y = y[:, 0]
    elif y.ndim != 1:
        raise ValueError("y must be a 1D target array.")
    return np.ascontiguousarray(y, dtype=np.float64)


def _as_multi_target(y):
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim != 2:
        raise ValueError("y must be a 2D target array with shape (n_samples, n_outputs).")
    return np.ascontiguousarray(y, dtype=np.float64)


def _as_eval_set(eval_set, target_converter):
    if eval_set is None:
        return None
    if not isinstance(eval_set, tuple) or len(eval_set) != 2:
        raise ValueError("eval_set must be a tuple of (X, y).")
    x_eval, y_eval = eval_set
    return _as_feature_matrix(x_eval), target_converter(y_eval)


class _WrapperMixin:
    def close(self):
        booster = getattr(self, "booster_", None)
        if booster is not None:
            booster.close()
        if hasattr(self, "booster_"):
            del self.booster_

    def dump(self, path):
        check_is_fitted(self, "booster_")
        self.booster_.dump(path)

    def __del__(self):
        self.close()


class SingleOutputGBDTRegressor(_WrapperMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        num_rounds=10,
        loss=b"mse",
        max_depth=4,
        max_leaves=32,
        max_bins=32,
        seed=0,
        num_threads=2,
        min_samples=20,
        lr=0.2,
        base_score=0.0,
        reg_l1=0.0,
        reg_l2=1.0,
        gamma=1e-3,
        early_stop=0,
        verbosity=Verbosity.SILENT,
        hist_cache=16,
        lib=None,
    ):
        self.num_rounds = num_rounds
        self.loss = loss
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bins = max_bins
        self.seed = seed
        self.num_threads = num_threads
        self.min_samples = min_samples
        self.lr = lr
        self.base_score = base_score
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.gamma = gamma
        self.early_stop = early_stop
        self.verbosity = verbosity
        self.hist_cache = hist_cache
        self.lib = lib

    def _native_params(self):
        return {
            "loss": self.loss,
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "max_bins": self.max_bins,
            "seed": self.seed,
            "num_threads": self.num_threads,
            "min_samples": self.min_samples,
            "lr": self.lr,
            "base_score": self.base_score,
            "reg_l1": self.reg_l1,
            "reg_l2": self.reg_l2,
            "gamma": self.gamma,
            "early_stop": self.early_stop,
            "verbosity": self.verbosity,
            "hist_cache": self.hist_cache,
        }

    def fit(self, X, y, eval_set=None):
        x_train = _as_feature_matrix(X)
        y_train = _as_single_target(y)
        if len(x_train) != len(y_train):
            raise ValueError("X and y must contain the same number of rows.")

        native_eval_set = _as_eval_set(eval_set, _as_single_target)
        self.close()
        self.booster_ = SingleOutputGBDT(lib=self.lib, params=self._native_params())
        self.booster_.set_data((x_train, y_train), native_eval_set)
        self.booster_.train(self.num_rounds)
        self.n_features_in_ = x_train.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "booster_")
        x = _as_feature_matrix(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but this model was fitted with "
                f"{self.n_features_in_} features."
            )
        return self.booster_.predict(x)


class MultiOutputGBDTRegressor(_WrapperMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        num_rounds=10,
        loss=b"mse",
        max_depth=4,
        max_leaves=32,
        max_bins=32,
        topk=0,
        one_side=True,
        seed=0,
        num_threads=2,
        min_samples=20,
        lr=0.2,
        base_score=0.0,
        reg_l1=0.0,
        reg_l2=1.0,
        gamma=1e-3,
        early_stop=0,
        verbosity=Verbosity.SILENT,
        hist_cache=16,
        lib=None,
    ):
        self.num_rounds = num_rounds
        self.loss = loss
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bins = max_bins
        self.topk = topk
        self.one_side = one_side
        self.seed = seed
        self.num_threads = num_threads
        self.min_samples = min_samples
        self.lr = lr
        self.base_score = base_score
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.gamma = gamma
        self.early_stop = early_stop
        self.verbosity = verbosity
        self.hist_cache = hist_cache
        self.lib = lib

    def _native_params(self):
        return {
            "loss": self.loss,
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "max_bins": self.max_bins,
            "topk": self.topk,
            "one_side": self.one_side,
            "seed": self.seed,
            "num_threads": self.num_threads,
            "min_samples": self.min_samples,
            "lr": self.lr,
            "base_score": self.base_score,
            "reg_l1": self.reg_l1,
            "reg_l2": self.reg_l2,
            "gamma": self.gamma,
            "early_stop": self.early_stop,
            "verbosity": self.verbosity,
            "hist_cache": self.hist_cache,
        }

    def fit(self, X, y, eval_set=None):
        x_train = _as_feature_matrix(X)
        y_train = _as_multi_target(y)
        if len(x_train) != len(y_train):
            raise ValueError("X and y must contain the same number of rows.")

        native_eval_set = _as_eval_set(eval_set, _as_multi_target)
        self.close()
        self.booster_ = MultiOutputGBDT(
            lib=self.lib,
            out_dim=y_train.shape[1],
            params=self._native_params(),
        )
        self.booster_.set_data((x_train, y_train), native_eval_set)
        self.booster_.train(self.num_rounds)
        self.n_features_in_ = x_train.shape[1]
        self.n_outputs_ = y_train.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "booster_")
        x = _as_feature_matrix(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but this model was fitted with "
                f"{self.n_features_in_} features."
            )
        return self.booster_.predict(x)


__all__ = [
    "SingleOutputGBDTRegressor",
    "MultiOutputGBDTRegressor",
]
