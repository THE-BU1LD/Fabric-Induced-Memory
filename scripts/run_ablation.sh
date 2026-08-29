#!/usr/bin/env bash

set -euo pipefail

cat >&2 <<'EOF'
ERROR: the legacy component-ablation wrapper has been disabled.

The previous version called a nonexistent root `run_train.py` and passed
`--no_memory`, `--no_retrieval`, `--hidden_dim`, and `--noise` flags that are
not implemented by the maintained `fim_experiments/main.py` CLI. Running that
script could therefore not constitute auditable ablation evidence.

Supported model-comparison smoke/suite execution is available through:

  python scripts/run_full_suite.py --device auto

Do not relabel model comparisons as component ablations.

Before paper-facing component ablations are run, implement explicit tested
configuration switches for each mechanism, freeze their semantics/protocol,
and preserve current-commit per-run provenance. See RESEARCH_TRUTH.md and the
open reproduction issue.
EOF

exit 2
