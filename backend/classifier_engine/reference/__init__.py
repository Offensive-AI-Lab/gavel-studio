"""Reference implementation of GAVEL (a verbatim copy of the upstream research code).

The files in this directory use absolute imports of the form
`from gavel.evaluation.metrics import compute_triggers`. Rewriting every
such import would make refreshing this copy from a newer upstream revision
painful, so instead we register this package's namespace as `gavel.*` in
sys.modules. After `import classifier_engine.reference`, every
`from gavel.X.Y import Z` inside this code resolves here.

This is a standard sys.modules namespacing trick for bundling a third-party
package under a local name. The mapping is established once at package-import
time. Pre-importing each subpackage guarantees the alias entries exist BEFORE
any module reaches its first `from gavel.X import ...` line.

To refresh this copy from a newer version of the reference implementation:
  1. Copy gavel/{evaluation,training,preprocessing,utils,models,config.py}
     into this directory verbatim
  2. Run the integration tests
No code edits inside this tree are needed.
"""
import importlib
import sys

# Step 1: alias this package itself as `gavel`.
sys.modules["gavel"] = sys.modules[__name__]

# Step 2: alias the subpackages our adapter actually uses.
# `evaluation/__init__.py` triggers `from gavel.evaluation.calibration import
# ...`, which means `gavel.evaluation` must already point at
# `reference.evaluation` BEFORE evaluation/__init__.py runs. So we pre-import
# in dependency order.
#
# We deliberately DO NOT eager-import `models` and `training` here:
# they pull in transformers at module-load time, and our integration
# tests + adapter code don't need them. If something later imports
# `from gavel.models import X`, Python's normal import machinery will
# still find the submodule via `gavel.__path__` (gavel is aliased to
# this package, which has `models/` as a subdirectory).
_SUBPACKAGES = ["config", "utils", "preprocessing", "evaluation"]

for _sub in _SUBPACKAGES:
    _full = f"{__name__}.{_sub}"
    _mod = importlib.import_module(_full)
    sys.modules[f"gavel.{_sub}"] = _mod
