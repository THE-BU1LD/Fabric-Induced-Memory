# Evidence integrity repair v1

Six reproduced evidence failures now have tested corrections in a complete, dependency-free reference validator. The repair is isolated from frozen studies. It is not yet integrated with the existing model runners, and it does not establish model efficacy or advance the scientific checkpoints.

| Retained failure | Correction in this package | Regression evidence |
|---|---|---|
| V01: repeated rows can satisfy a decision's counts | Exact prospective identities, unique records, complete cell/seed/sample coverage | Duplicate rows, cells, seeds and planned samples all rejected |
| V02: excessive family regression ignored | Family gains derived from paired raw vectors and each regression checked | One family losing 100% cannot pass despite four 50% wins |
| V03: NaN/Inf can pass metric comparison | Strict real-number types, finite inputs/intermediates and exact claim-key coverage | Nonfinite values on either side, overflow and malformed claims rejected |
| V04: incompatible L2 reductions | One sample metric, explicit origin/trajectory/seed/cell hierarchy | Unequal-energy example returns 0.5, not pooled 0.099504 |
| V05: trajectory counts change macro weights | Equal-cell reduction in the point estimate and each family-bootstrap draw | One trajectory versus nine still yields macro gain 0.5 |
| V06: failed/unmatched rows silently disappear | Complete planned set required before any reduction | A single failed or absent row produces an integrity failure with retained identity |

Additional checks reject unequal targets between methods, wrong source/data/plan identities, supplied win flags, and an independence boolean. Refusal receipts are retained and CLI failures exit nonzero. Receipt publication is atomic and never replaces an existing path.

The final source is covered by 43 adversarial tests; the last test receipt is [TEST_ROUND_3.json](TEST_ROUND_3.json). Earlier test receipts and exact source snapshots are preserved. Tests use software fixtures only. They imported no tensor/model packages. This is local author verification, not independent review. The full contract and correctness arguments are in [CONTRACT.md](CONTRACT.md).

## Use

Run the bounded regression suite from the repository root, choosing a fresh receipt path:

```sh
rtk proxy python3 -B successor/itfm_v4/repairs/evidence_integrity_v1/run_tests.py --receipt /tmp/fim-integrity-tests-new.json
```

The validator CLI takes `--plan`, `--plan-sha256`, `--evidence` and `--receipt`. The plan identity must come from the separate study freeze. Its exact schema is validated in `validate_plan`; the contract explains every field and reduction. `--help` lists the interface. Exit 0 means integrity checks completed; inspect `numerical_gates_pass` separately. Exit 2 means refused evidence, with a failure receipt when the output is writable. Exit 3 means receipt publication was refused or unavailable. No exit code authorizes scientific claims.

Both inputs are bounded to 8 MiB. Larger data requires a profiled streaming implementation. The CLI verifies identity consistency, not the truth of provenance declarations or an external operator's identity. Bootstrap intervals are exploratory; a single family yields no interval. Confirmation, multiplicity and independent reproduction require an admitted scientific protocol and external evidence.

## What remains

V07, the tensor writer's ignored observation mask, remains open. Its correction needs the observed-event contract and tensor noninterference tests, not a relabeling of this scalar verifier. Other unresolved work includes the research contribution, full model specification, real dataset and upstream license admission, correct learned memory behavior, complete baseline reproduction, protected confirmation, paper completion and independent verification.

The latest resource check failed at 81.48/109.47 MiB against the 512 MiB minimum, with two additional pageouts. No model or scientific campaign was launched. Earlier FIM/v2 results and the consumed v3 allowance remain unchanged.

The original 35-item checklist is preserved as the dated audit. [REPAIR_DISPOSITION.json](REPAIR_DISPOSITION.json) records what this implementation addresses and which integration/evidence obligations remain open. A reference fix is not recorded as a completed production or scientific integration.
