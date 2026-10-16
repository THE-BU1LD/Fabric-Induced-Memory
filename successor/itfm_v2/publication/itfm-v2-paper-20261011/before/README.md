# Paper package

`main.tex` is a complete evidence-bound manuscript for the frozen synthetic
ITFM v2 mechanism study. It intentionally does not report the unexecuted public
ITFM-R campaign as a result.

## Evidence boundary

- Supported: bounded synthetic causal reuse under the exact frozen construction.
- Not evaluated: public-data benefit, natural-system generalization, SOTA,
  external baseline superiority, and independent reproduction.
- Not release-certified: the current source has not been compiled or visually
  inspected under an admitted resource snapshot.

## Build and verify

From `successor/itfm_v2`, after the resource guard passes:

```bash
python3 scripts/build_paper_release.py verify
python3 scripts/build_paper_release.py stage --output output/arxiv/itfm-v2
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The Codex built-in LaTeX compiler may be used instead of `latexmk`. A release
requires successful compilation, page rendering, visual inspection of every
page, and a second `verify` against the final source. Do not place generated PDF
or auxiliary build files in the frozen result directory.

## Files

- `main.tex`: manuscript.
- `references.bib`: cited prior work.
- `generated_results.tex`: table transcribed from retained seed JSON artifacts.
- `CLAIM_EVIDENCE_MAP.md`: direct claim-to-artifact mapping and immutable hashes.
- `SOURCE_VERIFICATION_2026-10-11.json`: 19-check standard-library source and
  evidence-binding receipt.
- `../scripts/build_paper_release.py`: fail-closed evidence and package verifier.

## Submission decision

The source is suitable for internal scientific review now. It is not yet an
arXiv-ready release because PDF compilation/visual QA, authorship metadata,
license review, public confirmation disposition, and external reproduction are
open gates. Submitting the bounded synthetic study before public confirmation
would require an explicit decision to publish that narrower contribution.
