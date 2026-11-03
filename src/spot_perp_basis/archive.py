"""Archived-import alias; maintained implementation: data.ingest."""
import sys as _sys
from .data import ingest as _implementation
_sys.modules[__name__] = _implementation
