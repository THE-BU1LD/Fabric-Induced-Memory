# Change and evidence ledger — 16 October 2026

Base checkout HEAD: `32d8aa0b109f08a8ecd23c7bd97eaf29b5fe8989`.
The starting worktree already contained the historical `fim/` deletions, relocated
`FIM/`, untracked successors/audits and a modified README. It was neither a clean
release nor solely the old HEAD's source. `audits/kaggle-readiness-20261016/BEFORE.json`
records the initial status and exact pre-edit hashes; `before/` retains the modified
public-stack source files and README. No historical result or protocol was edited.

| Defect or requirement | Implemented change | Evidence boundary |
|---|---|---|
| Trainer constructor made its own exclusive run start fail | Delay logger/output creation until validated `fit` | Regression authored; numeric execution blocked |
| Partial accumulation and unequal validation batches biased reductions | Weight scalar per-example means by actual examples; final windows use their real denominator | Reference-update comparison authored |
| Resume repeated work and lost selection/RNG/counters | Schema-2 safe checkpoint, deterministic cursor, mutable-state contract, cumulative budgets, separate best/latest | Interrupted-vs-uninterrupted test authored |
| Unsafe checkpoint fallback | No unrestricted pickle fallback in trainer, backbone or bank loading | Source inspected; safe-load tests authored |
| Public writer stored hidden target coordinates | Explicit observation mask required; only fully visible mature rows enter the full-value bank | Hidden-coordinate mutation regression authored |
| No executable bounded fixture tied to public components | Extend the existing CLI and trainer; reuse MLP and TensorMemoryBank | Command/dry-run checks executed; model execution blocked |
| Repeated expensive resource polling | Amortize training admission per epoch or 30 seconds, with pre-import checks retained | Source reviewed; numeric frequency test authored |
| Failure or filename could masquerade as reusable completion | Exact source/config/data/raw artifact checks, source byte snapshots, exclusive outputs and refusal of partial attempts | Control-plane regressions executed |
| Quota loss on interruption or supervisor termination | Durable budget reservations; foreground bounded worker, graceful stop then owned-group cleanup | Simulated budget/timeout tests executed |
| Logs and analysis disconnected from model work | Training/evaluation JSONL, raw data/predictions, complete-inventory verified summaries | Log/export numerical assertions authored; archive hash tests executed |

No new architecture was introduced. The fixed retrieval diagnostic is explicitly
separate from correctness repairs. Encoder learning through detached retrieval is
not claimed. Per-owner capacity alone does not bound a generic endless stream;
this runner's finite owner inventory and allocation caps do. Whole-bank hashing
is kept out of its hot prediction path. AMP, compilation and multiprocessing are
not performance claims; runtime comparisons remain unmeasured.

The historical `FIM/scripts/run_full_suite.py` is a 25-cell **smoke** suite:
one epoch, 16 examples, batch 8, train rollout 4/eval rollout 8, seed 42. Its general
evaluation disables retrieval/writes; it cannot establish online memory efficacy.
The executed class is `FIMSystem`, not every richer component described elsewhere.
These interpretation limits do not erase the retained negative results.

Primary-source review was deliberately brief: [MemNO](https://arxiv.org/html/2409.02313v2)
already combines memory and Fourier operators; [Zoology](https://arxiv.org/abs/2312.04927)
warns against extrapolating easy recall fixtures; [FreshCast](https://arxiv.org/abs/2610.07834)
already studies refreshed residual memory; the [delay-aware adaptation audit](https://arxiv.org/abs/2610.12232)
motivates release-time controls and cheap online correctors. These were source
readings, not reproduced baselines or an exhaustive novelty review. A future
public study still needs equal available information, simple correctors, credible
long-history baselines and prospectively admitted data/protocols.

Final lightweight verification: 23 control tests, 13 AST/source compilation checks,
focused Ruff lint, CLI help and complete manifest dry-run passed. The actual CPU
tensor-test command exited nonzero before import; its receipt records zero tests.
No GPU checks ran. See `audits/kaggle-readiness-20261016/FINAL_VERIFICATION.json`
from the repository root for exact identities, commands and artifacts.
