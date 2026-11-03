"""Archived-import alias; maintained implementation: data.dataset."""
import sys as _sys
from .data import dataset as _implementation
_sys.modules[__name__] = _implementation
