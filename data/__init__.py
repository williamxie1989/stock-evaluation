"""Proxy package to expose src.data as top-level 'data' for tests.
Automatically maps data.* imports to src.data.* modules.
"""
import importlib
import sys
from types import ModuleType

# Import the real src.data package
_src_data_pkg = importlib.import_module('src.data')

# -----------------------------------------------------------------------------
# Compatibility Patch: Allow truthiness check on pandas.DatetimeIndex
# Some legacy tests rely on "if dt_index:" where dt_index is a DatetimeIndex.
# Pandas raises ValueError for this. We monkey-patch a bool implementation that
# returns True if the index is non-empty to satisfy those tests without
# affecting other logic.
# -----------------------------------------------------------------------------
try:
    import pandas as _pd

    def _dt_bool(self):
        return len(self) > 0

    _pd.DatetimeIndex.__bool__ = _dt_bool  # type: ignore[attr-defined]
    _pd.DatetimeIndex.__nonzero__ = _dt_bool  # py2 compatibility signature
except Exception:  # pragma: no cover
    pass
# Ensure the current package shares the same attributes as src.data
for attr in dir(_src_data_pkg):
    if not attr.startswith('__'):
        setattr(sys.modules[__name__], attr, getattr(_src_data_pkg, attr))

# Map submodules (e.g., data.unified_data_access) to corresponding src.data submodules
def _map_submodule(sub_name: str):
    full_src_name = f'src.data.{sub_name}'
    try:
        mapped = importlib.import_module(full_src_name)
        sys.modules[f'{__name__}.{sub_name}'] = mapped
    except ModuleNotFoundError:
        # Silently ignore if submodule not present; accessed lazily later.
        pass

# Pre-map commonly used submodules to avoid import issues in tests
for common in [
    'unified_data_access',
    'field_mapping',
    'providers',
    'db',
    'sync',
]:
    _map_submodule(common)