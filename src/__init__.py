"""Multi-agent sycophancy experiments: modular package refactored from experiment.ipynb."""

import os as _os

# ---------------------------------------------------------------------------
# Environment fix: keep transformers from probing the broken system
# tensorflow installation. ``/usr/lib/python3/dist-packages/tensorflow`` on
# this machine is a stale build compiled against an older numpy and raises
# ``Unable to convert function return value to a Python type`` inside
# ``dtypes.py`` as soon as it's imported. Transformers >= 4.57 only imports
# tensorflow lazily through ``image_transforms``/``processing_utils`` when
# ``USE_TF`` is not explicitly disabled. Setting ``USE_TF=0`` here (before
# any transformers import further down the module graph) makes it skip the
# probe and load just the torch path.
# ---------------------------------------------------------------------------
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
del _os

# ---------------------------------------------------------------------------
# Environment fix: the system ``/usr/lib/python3/dist-packages/matplotlib-
# 3.5.1-nspkg.pth`` runs at Python startup and pre-seeds
# ``sys.modules['mpl_toolkits']`` with the outdated system copy whose
# ``mplot3d`` imports ``matplotlib.docstring`` (removed in mpl >= 3.6). That
# stale cache entry survives even after ``sys.path`` is cleaned up, so
# matplotlib 3.10.x's ``projections/__init__.py`` fails to import Axes3D and
# the 3D projection (needed for LDA Fig 3) is never registered.
#
# Popping the cache here forces matplotlib (imported downstream) to re-resolve
# ``mpl_toolkits`` via ``sys.path`` and pick up the correct user-local copy at
# ``~/.local/lib/python3.10/site-packages/mpl_toolkits``.
# ---------------------------------------------------------------------------
import sys as _sys

for _m in [k for k in list(_sys.modules) if k == "mpl_toolkits" or k.startswith("mpl_toolkits.")]:
    del _sys.modules[_m]
del _sys
