"""Production-facing ITFM-R package.

This package is deliberately separate from :mod:`fim_v2`, which remains the
frozen synthetic mechanism study.  Importing :mod:`fim_sota` does not import
Torch so configuration, provenance, and resource checks can run first.
"""

from .config import CURRENT_CONFIG_VERSION, RunConfig, load_config

__all__ = ["CURRENT_CONFIG_VERSION", "RunConfig", "load_config"]

__version__ = "0.1.0"
