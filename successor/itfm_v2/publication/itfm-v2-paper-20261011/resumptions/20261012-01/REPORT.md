# ITFM v2 publication resumption — 12 October 2026

Checked at 2026-10-12T06:48:15.675212+05:30. **Publication remains blocked; no numerical replay or PDF compilation was admitted.**

## Work performed

- Invoked the existing isolated replay wrapper with a fresh output directory and unchanged 300-second/one-thread/three-seed budget. It stopped before copying the frozen source or launching a model.
- Checked PDF build admission separately. It failed before source copying, compiler launch or rendering. Tectonic is installed at `/opt/homebrew/bin/tectonic`.
- Re-ran the standard-library source, preserved-file, manuscript and scalar-record verifier: **228/228 PASS**, including **57/57 protected files unchanged** and **20/20 exact executed source files**.
- Rechecked the release gate: **nine substantive gates remain open**. No gate was promoted and no scientific result or manuscript was modified.
- Checked checkout identity, pre-existing dirty state, disk and process ownership. No competing ITFM runner or local lock was found; the unrelated LGWM campaign was left untouched. External-drive free space was 244.16 GiB. Git HEAD remains `32d8aa0b109f08a8ecd23c7bd97eaf29b5fe8989`; the untracked successor is still bound by artifact hashes, not a new commit.

## Fresh resource evidence

| Admission | First free + speculative | Second free + speculative | Pageout increase | Swap used (first → second) |
|---|---:|---:|---:|---:|
| Isolated replay | 158.39 MiB | 143.09 MiB | 1 | 10234.69 → 10234.69 MiB |
| PDF build | 146.67 MiB | 152.31 MiB | 2 | 9994.69 → 9994.69 MiB |

Both admissions fail the unchanged 512 MiB minimum and stable-pageout requirement. No guard was relaxed, no other process was stopped, and no dependency installation was attempted.

## Evidence

- [Replay receipt](../../replays/attempt-20261012-01/REPLAY_RECEIPT.json)
- [Build admission](BUILD_ADMISSION.json)
- [Preflight](PREFLIGHT.json)
- [Fresh verification](VERIFICATION.json)
- [Release gate](RELEASE_GATE.json)

## Remaining work

Once stable resources admit execution, use the existing isolated replay and PDF wrappers with new output directories. A successful same-host replay would still not satisfy a tested clean dependency environment or independent reproduction. After a successful PDF build, render and inspect every page. Independent scientific review/reproduction, immutable release identity/archive, distribution terms and destination remain separate open gates. Public benchmarking remains deferred and the predecessor remains closed.

This session changed only admission/verification receipts and status routing. Earlier failed receipts, the frozen protocol, retained outcomes, source archive and manuscript were preserved. Historical tooling-test results were not rerun or represented as fresh numerical evidence.
