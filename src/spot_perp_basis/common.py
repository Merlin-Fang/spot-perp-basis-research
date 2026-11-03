"""Archived-import alias; maintained implementation: artifacts."""
import sys as _sys
from . import artifacts as _implementation
_sys.modules[__name__] = _implementation
