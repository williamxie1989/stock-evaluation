"""Sitecustomize to patch pandas.DatetimeIndex boolean evaluation for legacy code.

Python automatically imports this module if it is found on sys.path during startup.
This patch ensures that evaluating a non-empty pandas.DatetimeIndex in a boolean
context returns True instead of raising a ValueError, restoring the behavior that
prior versions of pandas allowed and that some legacy code/tests rely on.
"""

import pandas as _pd

# -----------------------------------------------------------------------------
# Compatibility patch for pandas DatetimeIndex truthiness
# -----------------------------------------------------------------------------

def _dt_bool(self):  # type: ignore[override]
    """Return True if the DatetimeIndex is non-empty.

    This mirrors the behavior that pandas allowed prior to version 2.0. Newer
    pandas versions explicitly disallow truth-value testing of Index objects.
    Several legacy helpers/tests within this codebase rely on the previous
    behavior (e.g., `if missing_dates:` checks). By monkey-patching both the
    `__bool__` and the Python 2 alias `__nonzero__`, we retain backward
    compatibility while keeping the change well-scoped.
    """

    return len(self) > 0

# Only patch once to avoid recursion if already applied
if not getattr(_pd.DatetimeIndex, "__bool__", None) is _dt_bool:
    _pd.DatetimeIndex.__bool__ = _dt_bool  # type: ignore[assignment]
    _pd.DatetimeIndex.__nonzero__ = _dt_bool  # type: ignore[attr-defined]