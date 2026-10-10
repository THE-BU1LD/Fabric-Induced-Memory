# Research session entry contract

The single canonical state is `research/RESEARCH_STATE.json`, introduced in
[PR #30](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/30). This independent numerical
development branch does not duplicate that ledger. When the local canonical file
is absent, read its
[pinned state snapshot](https://github.com/THE-BU1LD/Fabric-Induced-Memory/blob/1c2de9851b6d6858e25f6cdedb641ef9300343e2/research/RESEARCH_STATE.json)
and check that PR for subsequent state updates. After integration, read the local
canonical file and preserve its accumulated history.

Before research or implementation work, read that state, `RESEARCH_TRUTH.md`,
and the applicable protocol. The state indexes historical sources; it does not
replace their evidence or certify that their full history has been imported.

Before finishing, append the exact source identity, verification, failures,
decisions and remaining gates to the same canonical file on its owning branch,
or to its integrated successor. Record independently developed work with an
immutable source-bound link until it is integrated. Do not initialize a competing
state file or discard an earlier result while reconciling branches.

This branch's bounded engineering record is
[`research/development/SDE_QUADRATIC_VARIATION_V2.json`](research/development/SDE_QUADRATIC_VARIATION_V2.json).
It is indexed by the pinned canonical snapshot above. Engineering verification
does not establish scientific efficacy, change a protected evaluation boundary,
or advance a research checkpoint. Preserve failed, negative, mixed and invalid
evidence; use a separately identified version and prospective protocol for any
scientifically meaningful successor.
