# Admission boundary

This directory records why the public ITFM-R campaign is blocked before stage
S1. `S0_STATUS.json` is a negative admission receipt, not a template and not an
authorization to fill unknown hashes with synthetic values.

S0 can change to `PASS` only when all of the following exist as independent,
hash-bound artifacts:

1. locked and locally verified upstream source identities;
2. authoritative source-license and dataset-license receipts;
3. dataset shard hashes, byte counts, trajectory-level splits, preprocessing
   source hash, normalization statistics, and official equivalence check;
4. named accelerator identity and maximum wall-clock, energy, and monetary
   budgets;
5. two stable host-resource samples above the frozen 512 MiB floor with no
   pageout increase and no conflicting heavy job; and
6. non-overwriting stage specifications whose commands and expected outputs are
   bound to those admitted identities.

Until then, no `admission/stages/*.json` or `admission/datasets/*.json` is
created, because dummy commands, fake hashes, or schema-only fixtures must not be
mistaken for executable scientific inputs.
