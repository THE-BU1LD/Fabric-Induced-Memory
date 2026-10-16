"""Load the exact retained evidence validator without tensor dependencies."""

import hashlib
import importlib.util
from pathlib import Path

VALIDATOR_SHA256 = "519bea984bb910426fd21ca98865777d9b47dbd3bcb4b67445ca74e8c8ea4832"
path = Path(__file__).resolve().parent.parent / "evidence_integrity_v1/evidence.py"
if hashlib.sha256(path.read_bytes()).hexdigest() != VALIDATOR_SHA256:
    raise RuntimeError("Evidence validator identity changed; integration must be reviewed")
spec = importlib.util.spec_from_file_location("itfm_retained_integrity_v1", path)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
