# Security and privacy review boundary

Status: **open; release blocked**.

- Dataset downloads require HTTPS, a declared byte count, SHA-256, and an
  admitted license receipt. Existing files are never overwritten.
- Dataset manifests reject path traversal, cross-split trajectory ownership,
  unregistered preprocessing, and unverified source hashes.
- Retrieval telemetry contains counts, similarity, age, latency, and memory
  only. It does not serialize protected targets.
- Checkpoints and memory banks can contain learned information and residual
  tensors. They are research artifacts and must not be published before data
  rights and membership-inference review.
- ERA5 is public environmental data, but its exact source terms, attribution,
  preprocessing, redistribution rights, and any joined metadata remain to be
  recorded in a signed admission receipt.
- External stage commands execute only from explicit, hash-bound stage specs;
  their stdout, stderr, exit code, inputs, outputs, and resource gate are kept.

No security, privacy, license, or release approval is implied by passing unit
tests or a local container build.
