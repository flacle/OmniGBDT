import importlib

from .lib_utils import Verbosity, load_lib
from .histogram import get_bins_maps
from .models import SingleOutputGBDT, MultiOutputGBDT
from .plotting import create_graph

__all__ = ["load_lib", "Verbosity", "create_graph", "get_bins_maps", "SingleOutputGBDT", "MultiOutputGBDT"]


def __getattr__(name):
    if name in {"SingleOutputGBDTRegressor", "MultiOutputGBDTRegressor"}:
        try:
            module = importlib.import_module(".sklearn", __name__)
        except ImportError as exc:
            raise ImportError(
                "The sklearn-compatible OmniGBDT wrappers require the optional "
                "scikit-learn dependency. Install it with "
                "`pip install \"omnigbdt[sklearn]\"`, "
                "`uv add \"omnigbdt[sklearn]\"`, or `uv sync --extra sklearn`."
            ) from exc
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
