# Strict evidence JSON admission — 11 October 2026

This separate engineering session starts from FIM draft PR44 at
`a659bc621f38a527397ea7ae7a18f2f144e1c7b2`. Its closed integration budget and
scientific holds remain unchanged.

## Observed defect and fixed scope

The trajectory-isolated analysis CLI and delayed-recall successor verifier use
ordinary JSON decoding. Repeated object fields are reduced to their last value
before existing validators can detect contradictions. Nonstandard NaN/Infinity
tokens and floating exponent overflow can also enter ignored metadata fields.

Add one shared, dependency-free JSON-object reader that rejects duplicate keys
at every depth and nonfinite real numbers. Wire only these two maintained CLI
read boundaries to it. Preserve the exact valid document values, original
manifest-byte digest, method validators and numerical comparison functions.
The retired shared-minibatch validator is outside this work.

## Prospective verification

Run the same new CLI-admission regressions once against original source and once
against the repair. Inputs are constructed JSON, with semantic/backend and
comparison functions replaced by call-recording doubles where appropriate.
Retain original failures and candidate output. Run the existing dependency-free
evidence-publication and delayed-recall method-admission suites as neighbors.
Allow one further targeted run only if a concrete repair is required. No process
may exceed 60 seconds; no full Torch collection or package installation is needed.

Verify compilation, whitespace, exact parent identity and unchanged existing
files outside the two reader edits and appended canonical index. Retain the
predecessor canonical index before appending the actual session record. Ask for
independent source review before publishing the code candidate.

## Scientific boundary

No model fit, benchmark generation, retained outcome read, protected statistical
analysis, retained demo rerun, scientific workflow dispatch, paid compute,
main merge, manuscript or protocol update, or scientific completion claim.
All valid statistical formulas, scientific inputs and historical adverse results
remain unchanged. Rejecting ambiguous evidence is an engineering admission fix.
