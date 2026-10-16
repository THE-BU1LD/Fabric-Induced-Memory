# ITFM v2 manuscript revision

The manuscript is **A Controlled Study of Episodic Tensor Residual Reuse under Exact Context Aliasing**, with Ryan as the sole author. It reports route B: a fixed exact-lookup synthetic case study. The original FIM negative study remains separate; the public ITFM-R campaign is deferred, not silently counted as completed.

## Current status

The implementation/wording discrepancies have been corrected and the exact 20-file executed source has been recovered. The numerical tables are generated from all nine conditions and all three original seeds. Original result files and historical receipts remain unchanged. The revised manuscript is not yet publication-ready: clean-environment/raw-output replay, independent reproduction/review, PDF compilation/inspection, author metadata/rights, and the immutable public archive remain open.

The retained files named `predictions-*.json` contain errors and retrieval metadata, not raw target/prediction tensors. A local scalar-record pass must not be called independent reproduction. The new replay worker is prepared but remains numerically unexecuted because resource admission failed.

## Safe verification and build

Run from `successor/itfm_v2`:

```sh
rtk python3 scripts/build_paper_release.py verify
rtk python3 scripts/build_paper_release.py gate
```

`gate` returns exit 3 while required evidence is missing. It checks exact manuscript hashes and gate-specific receipts. The commands never overwrite historical results.

After the repository's two-sample memory, pageout/swap, disk, ownership and budget checks admit work:

```sh
rtk python3 publication/itfm-v2-paper-20261011/tools/reproduce.py execute \
  --output publication/itfm-v2-paper-20261011/replays/UNUSED_DIRECTORY
rtk python3 publication/itfm-v2-paper-20261011/tools/build_pdf.py \
  --output publication/itfm-v2-paper-20261011/builds/UNUSED_DIRECTORY
```

Both wrappers require a fresh directory. The build uses only `main.tex`, `references.bib`, and `generated_results.tex`, and stops after 120 seconds. Render and inspect every resulting page before recording visual approval. This multi-file document is not supported by the standalone single-file compiler without flattening its inputs. Do not run the frozen historical verifier/runner in the live result tree.

## Evidence and history

- [Claim map](CLAIM_EVIDENCE_MAP.md) describes exactly what each retained artifact supports.
- [Revision workspace](../publication/itfm-v2-paper-20261011/README.md) contains authentic recovered source, preservation hashes, citation audit, tested tools, blocked-attempt receipts and release gates.
- `SOURCE_VERIFICATION_2026-10-11.json` is the historical 19-check receipt for the earlier manuscript. It does not certify this revision. The earlier paper is preserved in `../publication/itfm-v2-paper-20261011/before/`.
- The comprehensive checklist and per-item disposition remain the authority for outstanding preparation work. No submission or public release has occurred.
