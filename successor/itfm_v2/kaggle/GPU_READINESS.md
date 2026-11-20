# Current readiness — 19 October 2026

**CPU engineering: LOCALLY_TESTED. GPU: NOT_EXECUTED. Scientific contribution:
NOT_ESTABLISHED.** Read the dated historical report below in that context.

Implementation commit `7b9c2af4251248eb155e07ab88ea00cbca46184d`, runtime source
SHA-256 `601db1e7acc5be34d20e1913225b3e9ec6d108a36dfdda8b73f2307e6d461291`:
52 numerical/runtime tests passed, four CUDA tests skipped; 23 standard-library
controls passed. A seed-101, two-epoch CPU fixture completed ten updates and
72 example presentations in 7.335284 seconds end to end. Best/latest checkpoints,
raw data, predictions, logs and a source snapshot are retained in
`audits/fim-causal-recovery-20261019/runs_02/cpu-smoke/` in the repository.

This host was Linux x86-64, Python 3.12.14, Torch 2.11.0+cpu, NumPy 2.4.4,
pytest 9.1.1, one thread and an 8 GiB cgroup. Actual double-sampled admission
passed before tensor execution. Earlier resource refusals and pre-fix failures
are preserved. The memory admission floor was not weakened.

No Kaggle account was authenticated in the available browser: Login/Sign Up
were visible. Accelerator, remaining quota and current jobs were not verified;
no Kaggle job ran. CUDA/FP16/BF16 correctness and CPU/GPU cost comparisons remain
unexecuted. The smallest next action is the explicit CUDA preflight and tiny
CUDA suite in HANDOFF.md, after actual host/account/job inspection.

The five-control smoke is not a benchmark claim: test MSE was .00835317594 for
full memory, .00806379867 for last residual and .00033732270 for persistence.
Full memory therefore did not win. Post hoc retained-prediction analysis found
93.0876% of full-memory squared error at cold start, but warm retrieval also
lost to the simpler controls. No consumed study or contribution gate changed.

The full report and current research state in the repository are authoritative
for this session; the October 16 report below is retained unchanged.

# GPU readiness — 16 October 2026

**Status: BLOCKED.** New numerical correctness, exact recovery and GPU precision
checks have not executed. The implementation, tests and launch package exist;
static/control-plane checks do not establish tensor correctness or GPU readiness.

## A. Engineering implemented and checked

The existing `fim-sota` CLI now connects strict preflight, real synthetic training,
safe checkpoints/resume, frozen evaluation, a finite manifest, analysis and export.
Trainer, writer and safe-loading defects were repaired. Historical source bytes
were retained first. Source/config/data identities, unique outputs, simulated-time
budgets, completion validation and allowlisted archive hashes have executable
standard-library tests. See `CHANGE_LEDGER.md` for the cause and repair mapping.

## B. CPU checks actually executed

23 standard-library control tests passed in the final integrated check. AST and
source compilation checks do not import Torch. The final audit receipt records
the final source/test hashes and checked help, manifest dry-run and lint. The real local preflight
refused execution before numerical imports: free-plus-speculative samples were
129,286,144 and 140,558,336 bytes (about 123.3/134.0 MiB), below 536,870,912 bytes.
Pageouts and swapouts were stable in that sample; low memory alone failed admission.
The final actual CPU test command also refused before import at 160,284,672 and
146,718,720 bytes (152.9/139.9 MiB), with zero tests executed and a nonzero exit.
The initial sandbox separately denied `ps` and `sysctl vm.swapusage`; `vm_stat`
supplied the memory/pageout/swapout counters used by the implemented guard.

Python 3.14.7, Torch 2.11.0 and NumPy 2.4.4 were read from package metadata. No
packages were changed. Pytest was absent. **Zero new numerical CPU tests or model
training runs passed.** Requested tensor checks retain a blocked receipt, not a
fabricated skip/pass. Older hosted v4 checks exercised different source and are
not substituted for this revision's tests.

## C. GPU checks actually executed

**None.** No CUDA import, CUDA training, AMP measurement, GPU profiler, cloud
submission or quota-consuming job was performed. There is no throughput estimate
or claim that this small workload benefits from a GPU.

## D. Remaining blockers and assumptions

1. Run the authored recovery/causality/memory/rollout tests on an admitted CPU
   environment, with pytest installed. Exact recovery is implemented but unverified.
2. Run explicit CUDA preflight and the tiny CUDA training/evaluation tests. Keep
   FP32 until actual hardware-specific AMP checks pass. Investigate any failure;
   do not weaken assertions or silently fall back to CPU.
3. Measure CPU versus GPU end-to-end time and memory before accepting the preliminary
   development manifest. Its four-hour ceiling is a safety bound, not a runtime promise.
4. The proposed generator is a bounded synthetic fixture. Public data terms,
   exact upstream baselines, meaningful contribution, protected protocol and
   independent reproduction remain unadmitted. This package cannot pass those gates.
5. Evaluation restarts from its frozen checkpoint; mid-stream online-bank recovery
   is not implemented. Generic models with undeclared mutable state are refused.
   Optional external dashboards are not connected; local logs are authoritative.

## E. Historical evidence excluded from current confirmation

The original negative FIM studies, v2 exact-residual synthetic findings and retained
replays, the incomplete consumed v3 campaign, and v4's previous hosted observation
tests remain in their original provenance. None is a performance/recovery/CUDA
result for these new bytes. V3 is not restarted, and v4's contribution gate is not
advanced. No superiority, publication-readiness or independent-reproduction claim
is made.

The smallest next GPU action is the CUDA preflight plus the tiny test command in
`HANDOFF.md`, after CPU correctness passes and cloud execution is separately
authorized. Do not launch the full manifest first.
