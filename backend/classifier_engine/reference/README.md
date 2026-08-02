# Reference implementation

This directory is a verbatim copy of the reference GAVEL research
implementation.

**Do not edit files here directly.** They are kept as an unmodified copy so
that our backend runs the exact same numerical algorithms the reference uses
(threshold sweep, Youden-J, use-case detection, AUC, sliding-window
inference). Any adaptations (loading data from our DB, integrating with
FastAPI) live in `backend/evaluation/adapter.py`, NOT here.

If you need to fix something in this code:
1. Patch it upstream and refresh this copy, OR
2. Override the behavior at the adapter layer.

This tree keeps its original absolute imports (e.g.
`from gavel.evaluation.calibration import calibrate`); `reference/__init__.py`
registers this package's namespace as `gavel.*` in `sys.modules` so those
imports resolve here without needing the upstream package installed.
