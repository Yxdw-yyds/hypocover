"""Load the official HypoSpace code (vendored, unmodified) as a library.

Each upstream domain folder has its own top-level package called ``modules``,
so the three domains cannot be imported side by side with a plain
``sys.path`` hack. We import one domain at a time and then drop the colliding
names from ``sys.modules``; the objects we keep hold their own references.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "HypoSpace"

# Sibling module names inside the upstream domain folders that would clash.
_CLASHING = ("modules", "generate_causal_dataset", "boolean_dataset",
             "generate_3d_dataset_complete")


def _purge() -> None:
    for name in list(sys.modules):
        if name.split(".")[0] in _CLASHING:
            del sys.modules[name]


@lru_cache(maxsize=None)
def load(domain: str, module: str) -> ModuleType:
    """Import ``module`` from upstream ``domain`` folder (causal | 3d | boolean)."""
    folder = UPSTREAM_ROOT / domain
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    _purge()
    sys.path.insert(0, str(folder))
    try:
        # Upstream modules print on import in places; keep our logs clean.
        with contextlib.redirect_stdout(io.StringIO()):
            mod = importlib.import_module(module)
    finally:
        sys.path.remove(str(folder))
    # Re-register under a unique name so later loads don't evict it.
    sys.modules[f"_hypospace_{domain}_{module}"] = mod
    _purge()
    return mod
