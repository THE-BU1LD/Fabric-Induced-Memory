# FIM fresh paired multi-seed replication protocol v2 — frozen 2026-09-08

## Chronology and why v2 exists

This is a separately versioned correction to the unmerged v1 draft in PR #8. The v1 draft was superseded **before any fresh outcome was inspected** because static audit found a baseline-pairing flaw in the maintained experiment order: model construction consumes the global RNG before synthetic static data generation, and evaluation later consumes the global RNG after architecture-dependent training. Different architectures therefore need not receive identical training/evaluation examples even when the same top-level seed is supplied. Transformer training also contains dropout, so later global-RNG state can diverge further.

No observed v1 result motivated this correction. Any v1 workflow artifact is engineering-only and must not be used as scientific evidence.

V2 keeps the scientific question, fresh seeds, benchmarks, model families, compute budget, primary metric, minimum meaningful effect, and advancement rule fixed. It changes only the RNG isolation required for fair paired comparisons and makes the FIM width consistent between the mechanism and baseline matrices.

## Known prior evidence

The retained 2026-08-29 three-seed current-code component matrix is already known:

- Lorenz-96: mixed/adverse to broad component-superiority claims;
- delayed recall: near-null, only weakly favorable numerically to the full model;
- `no_memory` and `no_retrieval`: identical on reported rollout metrics in that matrix.

V2 is therefore a fresh-seed replication/triage study, not a blind first experiment.

## Frozen source boundary

Protocol designed against canonical main SHA:

`68dcb42242ee6e11f2243dccd6f3e0f3460e1545`

Allowed v2 execution diff is limited to this protocol, the paired runner/test, and workflow orchestration. It must not alter model mathematics, benchmark equations, loss definitions, training objective, or outcome metrics.

## Frozen fresh seeds

`101, 202, 303, 404, 505`

No failed or unfavorable seed may be replaced.

## Frozen benchmarks

- `delayed_recall`
- `lorenz96`

## Paired RNG contract

For benchmark-specific offset:

- `lorenz96`: `10_000`
- `delayed_recall`: `20_000`

For each top-level seed `s`:

- model/training stochastic seed: `s`;
- synthetic dataset-generation seed: `1_000_000 + benchmark_offset + s`;
- train/validation split seed: `data_seed + 1`;
- DataLoader shuffle generator seed: `data_seed + 2`;
- evaluation initial-condition seed: `2_000_000 + benchmark_offset + s`.

Dataset generation and evaluation run inside isolated RNG contexts so they cannot be shifted by architecture-specific parameter initialization, dropout, or other model-specific random draws. The training DataLoader uses its own explicit generator.

The same benchmark + top-level seed therefore has the same generated static data, split, shuffle stream, and evaluation initial conditions across every compared arm.

## Frozen model dimensions and training budget

All maintained neural arms use:

- hidden width: `64`;
- FIM trace dimension: `32` where applicable;
- epochs: `6`;
- batch size: `64`;
- dataset size: `1024`;
- training rollout steps: `4`;
- evaluation rollout steps: `30`;
- workers: `0`;
- device in CI: `cpu`.

This matches data access and optimizer-step opportunity, not parameter count. Parameter counts are retained and must be reported. The bounded budget is a replication/triage budget, not a convergence or compute-matched SOTA study.

## Frozen mechanism matrix

Variants:

- `full`;
- `no_memory`;
- `no_retrieval`;
- `no_salience_gating`.

Expected cells: `2 benchmarks × 5 seeds × 4 variants = 40`.

The current-code semantic tests must pass before outcome-bearing execution. `no_retrieval` is a write-only structural control: it can store traces but cannot feed retrieved content into prediction. Performance identity with `no_memory` is therefore possible by construction; the scientifically relevant retrieval contrast is `full` vs `no_retrieval`.

## Frozen maintained-baseline matrix

Models:

- `fim_plus`;
- `transformer`;
- `ssm`;
- `deeponet`;
- `mlp`.

Expected cells: `2 benchmarks × 5 seeds × 5 models = 50`.

No baseline may receive less data or fewer epochs because it is competitive. Parameter counts are reported rather than assumed matched.

## Outcomes

Primary metric: **rollout MSE** (lower is better).

Secondary metric: rollout MAE.

### Primary retrieval-feedback replication

On `delayed_recall`, paired by fresh seed:

`delta = MSE(full) - MSE(no_retrieval)`.

The bounded advancement rule is unchanged from v1 and passes only if both hold:

1. mean relative rollout-MSE improvement of `full` over `no_retrieval` is at least **2%**; and
2. `full` has lower rollout MSE on at least **4 of 5** paired seeds.

Failure to pass is a valid null/adverse result and does not authorize tuning this protocol.

### Secondary contrasts

- `full` vs `no_salience_gating` per benchmark;
- `full` vs `no_memory` per benchmark;
- `fim_plus` vs Transformer, SSM, DeepONet, and MLP per benchmark.

With five seeds, report per-seed paired deltas, means, sample SDs, and direction counts. Do not use tiny-n significance tests to manufacture confidence.

## Failure policy

- Every attempted cell is written to the manifest as `success` or `failed`.
- Failed cells retain exception type/message/traceback and any already-written raw run artifacts.
- No failure is silently omitted.
- No replacement seed is allowed.
- Paper-facing aggregation is incomplete unless all 90 expected cells succeed with finite metrics.
- NaN/Inf metrics fail closed.

## Provenance

The retained manifest must record:

- exact executed Git SHA and dirty-tree status;
- Python/PyTorch/platform identity;
- protocol SHA-256;
- top-level/model seed, data seed, split/shuffle derivation, and evaluation seed;
- benchmark, model/variant, parameter count, config, start/end time, metric or failure state;
- unique result directory for every cell.

## Claim boundary

Permitted if supported:

- bounded fresh-seed replication statements for these two synthetic benchmarks and this exact budget;
- benchmark-specific positive/null/mixed/adverse findings;
- baseline wins;
- structural evidence that the ablation switches execute their documented paths.

Not permitted from v2 alone:

- broad FIM superiority;
- state-of-the-art claims;
- convergence or compute-efficiency superiority;
- historical-paper equivalence;
- external or real-world validation;
- treating workflow/CI success as scientific success.

## Canonical command

```bash
python scripts/run_paired_fresh_multiseed_v2.py
```

The script itself performs preflight pairing checks, executes all attempted cells, writes a fail-closed manifest, and generates `summary.json` plus `paper_evidence.md` only from retained row-level evidence.

## Freeze rule

After the first v2 outcome-bearing cell starts, the seeds, benchmarks, RNG formulas, variants, baselines, dimensions, training budget, metrics, primary contrast, and advancement rule are immutable. Any later correction or new hypothesis must be versioned separately.
