"""Deprecated compatibility module.

This project no longer uses `Globals.py` as the source of experimental
configuration. Dataset paths, checkpoint paths, training schedules, and
distillation hyperparameters should be specified through `configs/` and
`scripts/`.

This file is kept only to avoid import errors in legacy code paths.
"""

DEPRECATED_GLOBALS = True
